# trading_engine.py
import config
import logging
import numpy as np

logger = logging.getLogger("BacktestLogger")

class AccountManager:
    def __init__(self, balance):
        self.balance = balance
        self.position = None 
        self.peak_price = 0

    # [수정] genes 파라미터 추가: GA가 찾아낸 최적의 유전자 사용
    def calculate_trade_parameters(self, score, pred_volatility, current_price, genes):
        confidence = abs(score)
        
        # 1. 진입 장벽 동적 계산 (유전자 적용)
        # 수식: 기본장벽 + (변동성 * 민감도)
        # 변동성이 클수록 진입 장벽이 높아져서 뇌동매매 방지
        dynamic_threshold = genes['base_threshold'] + (pred_volatility * genes['vol_impact'])
        dynamic_threshold = min(dynamic_threshold, 0.9) # 최대 0.9 제한

        if confidence < dynamic_threshold:
            return 0, 0, 0, 0 # 진입 불가

        # 2. 레버리지 (기존 로직 유지)
        raw_leverage = 1 + (confidence * (config.MAX_LEVERAGE - 1)) / 0.8 
        leverage = int(np.clip(round(raw_leverage), config.MIN_LEVERAGE, config.MAX_LEVERAGE))
        
        # 3. 리스크 금액 계산
        risk_pct = confidence * config.MAX_RISK_PER_TRADE_CAP
        risk_amount = self.balance * risk_pct

        # 4. SL / TP 결정 (유전자 적용)
        # 수식: 변동성 * 손절배수(sl_mul)
        vol_range = max(pred_volatility, 0.005) # 최소 변동성 보정
        sl_dist = current_price * (vol_range * genes['sl_mul'])
        
        # 익절은 손절폭 대비 비율(tp_ratio)로 결정
        tp_dist = sl_dist * genes['tp_ratio']
        
        # 수량 계산
        if sl_dist == 0: qty = 0
        else: qty = risk_amount / sl_dist
        
        # 레버리지 한도 체크
        max_qty = (self.balance * leverage) / current_price
        qty = min(qty, max_qty)
        
        return leverage, qty, sl_dist, tp_dist

    def update_pnl_and_check_exit(self, current_close, current_high, current_low, timestamp, sl_price, tp_price):
        if not self.position: return 0
        
        entry = self.position['price']
        size = self.position['size']
        p_type = self.position['type']
        lev = self.position['leverage']
        
        # 펀딩비 차감 (간소화)
        position_value = current_close * size
        self.balance -= position_value * config.FUNDING_RATE_4H

        # 강제 청산 체크
        liq_threshold = 1.0 / lev
        is_liquidated = False
        
        if p_type == 'LONG':
            liq_price = entry * (1 - liq_threshold)
            if current_low <= liq_price: is_liquidated = True
        else: 
            liq_price = entry * (1 + liq_threshold)
            if current_high >= liq_price: is_liquidated = True
                
        if is_liquidated:
            logger.warning(f"!!! [LIQUIDATION] Price hit {liq_price:.2f} (Lev: x{lev})")
            loss = (entry * size) / lev 
            self.balance -= loss 
            self.position = None
            return -loss

        # 청산 로직
        close_signal = False
        reason = ""
        exec_price = current_close

        if p_type == 'LONG':
            self.peak_price = max(self.peak_price, current_high)
            if current_low <= sl_price:
                close_signal = True; reason = "AI_StopLoss"; exec_price = sl_price
            elif current_high >= tp_price:
                # Trailing Profit: 익절 구간 도달 후 20% 반납 시 청산
                trailing_gap = (tp_price - entry) * 0.2 
                if current_close <= (self.peak_price - trailing_gap):
                    close_signal = True; reason = "AI_TrailingProfit"
        
        elif p_type == 'SHORT':
            self.peak_price = min(self.peak_price, current_low)
            if current_high >= sl_price:
                close_signal = True; reason = "AI_StopLoss"; exec_price = sl_price
            elif current_low <= tp_price:
                trailing_gap = (entry - tp_price) * 0.2
                if current_close >= (self.peak_price + trailing_gap):
                    close_signal = True; reason = "AI_TrailingProfit"

        if close_signal:
            self._force_close(exec_price, timestamp, reason)
            return 0

        # 평가손익 반환
        if p_type == 'LONG': return (current_close - entry) * size
        else: return (entry - current_close) * size

    def _force_close(self, price, timestamp, reason):
        if not self.position: return
        
        if self.position['type'] == 'LONG': real_price = price * (1 - config.SLIPPAGE)
        else: real_price = price * (1 + config.SLIPPAGE)

        size = self.position['size']
        entry = self.position['price']
        
        if self.position['type'] == 'LONG': pnl = (real_price - entry) * size
        else: pnl = (entry - real_price) * size
            
        pnl -= (real_price * size * config.COMMISSION)
        self.balance += pnl
        
        logger.info(f"[{timestamp}] Close {self.position['type']} ({reason}) | Price: {real_price:.2f} | PnL: {pnl:+.2f} | Bal: {self.balance:.0f}")
        self.position = None
        self.peak_price = 0

    def execute_trade(self, signal, price, qty, leverage, sl_dist, tp_dist, timestamp):
        if self.position:
             if (self.position['type'] == 'LONG' and signal == 'OPEN_SHORT') or \
               (self.position['type'] == 'SHORT' and signal == 'OPEN_LONG'):
                self._force_close(price, timestamp, 'SignalSwitch')

        if signal in ['OPEN_LONG', 'OPEN_SHORT'] and not self.position:
            if self.balance <= 0 or qty <= 0: return 

            if signal == 'OPEN_LONG':
                real_entry = price * (1 + config.SLIPPAGE)
                p_type = 'LONG'
                sl_price = real_entry - sl_dist
                tp_price = real_entry + tp_dist
            else:
                real_entry = price * (1 - config.SLIPPAGE)
                p_type = 'SHORT'
                sl_price = real_entry + sl_dist
                tp_price = real_entry - tp_dist
            
            self.balance -= (real_entry * qty * config.COMMISSION)
            
            self.position = {
                'type': p_type, 
                'price': real_entry, 
                'size': qty, 
                'leverage': leverage,
                'sl_price': sl_price,
                'tp_price': tp_price
            }
            self.peak_price = real_entry
            logger.info(f"[{timestamp}] Open {p_type} @ {real_entry:.2f} (Qty: {qty:.4f}, Lev: x{leverage})")
            logger.info(f"    >> Params: SL {sl_price:.2f} / TP {tp_price:.2f}")