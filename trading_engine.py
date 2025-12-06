# trading_engine.py
import config
import logging
import numpy as np

logger = logging.getLogger("BacktestLogger")

class AccountManager:
    def __init__(self, balance, leverage):
        self.initial_balance = balance
        self.balance = balance
        self.base_leverage = leverage 
        self.position = None 
        self.peak_price = 0
        self.is_bankrupt = False
        self.trades_history = [] 

    # AI 점수(확신도)에 따른 동적 레버리지 계산
    def get_dynamic_leverage(self, score):
        confidence = abs(score)
        
        if confidence < config.ENTRY_THRESHOLD: return 1.0
        
        ratio = (confidence - config.ENTRY_THRESHOLD) / (1.0 - config.ENTRY_THRESHOLD)
        ratio = max(0.0, min(1.0, ratio))
        
        # 비선형 레버리지 적용 (확신도가 높을수록 레버리지를 급격히 증가시켜 수익성 극대화)
        nonlinear_ratio = ratio ** 2 
        
        dyn_leverage = config.MIN_LEVERAGE + (nonlinear_ratio * (config.MAX_LEVERAGE - config.MIN_LEVERAGE))
        
        return round(dyn_leverage, 1)

    # 진입 수량 계산 (리스크 관리 적용)
    def get_position_qty(self, price, score, atr, leverage):
        if self.is_bankrupt: return 0
        confidence = abs(score)
        
        # 리스크 자본 할당 (AI 확신도가 높을수록 더 많은 리스크 감수)
        risk_range = config.MAX_RISK_PER_TRADE - config.BASE_RISK_PER_TRADE
        target_risk_pct = config.BASE_RISK_PER_TRADE + (
            (confidence - config.ENTRY_THRESHOLD) / (1 - config.ENTRY_THRESHOLD) * risk_range
        )
        target_risk_pct = min(max(target_risk_pct, config.BASE_RISK_PER_TRADE), config.MAX_RISK_PER_TRADE)
        
        risk_amount = self.balance * target_risk_pct

        # 변동성 기반 수량 조절 (손절폭을 3 ATR로 가정하고 리스크 금액 역산)
        sl_distance = atr * 3.0
        if sl_distance == 0: return 0
        qty = risk_amount / sl_distance
        
        # 레버리지 한도 체크 (슬리피지 고려하여 최대 가능 수량 제한)
        price_with_slippage = price * (1 + config.SLIPPAGE)
        max_qty_by_leverage = (self.balance * leverage) / price_with_slippage
        
        return min(qty, max_qty_by_leverage)

    # 포지션 평가, 펀딩비 차감 및 청산 조건 확인
    def update_pnl_and_check_exit(self, current_close, current_high, current_low, timestamp, current_atr):
        if not self.position: return 0
        
        entry = self.position['price']
        size = self.position['size']
        p_type = self.position['type']
        lev = self.position['leverage']
        
        # 펀딩비 차감 (4시간마다 발생 가정)
        position_value = current_close * size
        funding_fee = position_value * config.FUNDING_RATE_4H
        self.balance -= funding_fee

        # 1. 강제 청산(Liquidation) 체크
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
            self.peak_price = 0
            return -loss

        # 2. 미실현 손익률(PnL %) 계산
        if p_type == 'LONG':
            pnl_pct = (current_close - entry) / entry
        else:
            pnl_pct = (entry - current_close) / entry

        # 3. 동적 청산 (손절 및 트레일링 스탑)
        sl_dist = current_atr * 3.0
        
        trailing_dist = None
        if pnl_pct > 0.05: trailing_dist = current_atr * 2.5 
        if pnl_pct > 0.15: trailing_dist = current_atr * 1.5 

        close_signal = False
        reason = ""
        exec_price = current_close

        if p_type == 'LONG':
            self.peak_price = max(self.peak_price, current_high)
            
            # 손절매 (Stop Loss)
            if current_low <= (entry - sl_dist):
                close_signal = True; reason = "StopLoss(ATR)"; exec_price = entry - sl_dist
            
            # 트레일링 스탑 (Trailing Profit)
            elif trailing_dist and current_close <= (self.peak_price - trailing_dist):
                if current_close > entry * 1.005: 
                    close_signal = True; reason = "TrailingProfit"
        
        elif p_type == 'SHORT':
            self.peak_price = min(self.peak_price, current_low)
            
            # 손절매 (Stop Loss)
            if current_high >= (entry + sl_dist):
                close_signal = True; reason = "StopLoss(ATR)"; exec_price = entry + sl_dist
            
            # 트레일링 스탑 (Trailing Profit)
            elif trailing_dist and current_close >= (self.peak_price + trailing_dist):
                if current_close < entry * 0.995:
                    close_signal = True; reason = "TrailingProfit"

        if close_signal:
            self._force_close(exec_price, timestamp, reason)
            return 0

        if p_type == 'LONG': return (current_close - entry) * size
        else: return (entry - current_close) * size

    # 포지션 강제 종료 (내부 함수)
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

    # 매매 실행 (진입 및 스위칭)
    def execute_trade(self, signal, price, qty, leverage, timestamp):
        # 이미 포지션이 있는 경우, 반대 신호나 청산 신호가 오면 기존 포지션 종료
        if self.position:
            should_close = False
            if self.position['type'] == 'LONG' and signal == 'OPEN_SHORT': should_close = True
            if self.position['type'] == 'SHORT' and signal == 'OPEN_LONG': should_close = True
            if signal == 'CLOSE': should_close = True
            
            if should_close:
                self._force_close(price, timestamp, 'SignalSwitch')

        # 신규 진입
        if signal in ['OPEN_LONG', 'OPEN_SHORT'] and not self.position:
            if self.balance <= 0: return 

            if signal == 'OPEN_LONG':
                real_entry_price = price * (1 + config.SLIPPAGE)
                p_type = 'LONG'
            else:
                real_entry_price = price * (1 - config.SLIPPAGE)
                p_type = 'SHORT'
                
            if qty <= 0: return
            
            # 증거금 확인 및 수량 조정
            initial_margin = (real_entry_price * qty) / leverage
            if initial_margin > self.balance:
                qty = (self.balance * leverage) / real_entry_price

            # 수수료 차감
            self.balance -= (real_entry_price * qty * config.COMMISSION)
            
            self.position = {
                'type': p_type, 
                'price': real_entry_price, 
                'size': qty, 
                'leverage': leverage 
            }
            self.peak_price = real_entry_price
            logger.info(f"[{timestamp}] Open {p_type} @ {real_entry_price:.2f} (Qty: {qty:.4f}, Lev: x{leverage})")
