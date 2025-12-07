# trading_engine.py
import config
import logging
import numpy as np

logger = logging.getLogger("BacktestLogger")

class AccountManager:
    def __init__(self, balance, leverage):
        self.initial_balance = balance
        self.balance = balance
        self.base_leverage = int(leverage)
        self.position = None 
        self.peak_price = 0
        self.is_bankrupt = False

    def get_dynamic_leverage(self, score, threshold):
        confidence = abs(score)
        if confidence < threshold: return 1
        
        ratio = (confidence - threshold) / (1.0 - threshold + 1e-9)
        ratio = max(0.0, min(1.0, ratio))
        
        raw_lev = config.MIN_LEVERAGE + (ratio * (config.MAX_LEVERAGE - config.MIN_LEVERAGE))
        return int(round(raw_lev))

    def get_position_qty(self, price, score, atr, leverage, risk_scale, sl_mult):
        if self.is_bankrupt or self.balance <= 0: return 0
        confidence = abs(score)
        
        target_risk_pct = min(risk_scale, config.GLOBAL_RISK_LIMIT)
        risk_amount = self.balance * target_risk_pct
        
        sl_distance = atr * sl_mult
        if sl_distance == 0: return 0
        
        qty_risk = risk_amount / sl_distance
        price_with_slippage = price * (1 + config.SLIPPAGE)
        max_qty_lev = (self.balance * leverage) / price_with_slippage
        
        return min(qty_risk, max_qty_lev)

    def check_exit_with_precision(self, candles_5m, entry, size, p_type, lev, sl_dist, tp_ratio):
        """
        [업데이트] 수익 극대화 트레일링 스탑 로직 적용
        """
        if candles_5m.empty: return False, 0, "", None

        liq_threshold = 1.0 / float(lev)
        if p_type == 'LONG':
            liq_price = entry * (1 - liq_threshold + 0.005) 
            current_stop_price = entry - sl_dist
        else:
            liq_price = entry * (1 + liq_threshold - 0.005)
            current_stop_price = entry + sl_dist
            
        # 트레일링 활성화 가격 (기존 고정 TP 가격)
        trailing_activation_price = (entry + (sl_dist * tp_ratio)) if p_type == 'LONG' else (entry - (sl_dist * tp_ratio))
        is_trailing_active = False
        
        # 최고점/최저점 추적
        best_price = entry 
        
        for ts, row in candles_5m.iterrows():
            curr_h = row['high']
            curr_l = row['low']
            
            # 1. 청산 체크 (최우선)
            if p_type == 'LONG' and curr_l <= liq_price: return True, liq_price, "LIQUIDATION", ts
            if p_type == 'SHORT' and curr_h >= liq_price: return True, liq_price, "LIQUIDATION", ts

            # 2. 트레일링 스탑 로직
            if p_type == 'LONG':
                if curr_h > best_price: best_price = curr_h
                
                # 목표 수익률 도달 시 트레일링 활성화
                if not is_trailing_active and curr_h >= trailing_activation_price:
                    is_trailing_active = True
                
                # 활성화 상태: 최고점 대비 0.5 * SL거리 만큼 여유를 두고 따라감
                if is_trailing_active:
                    new_stop = best_price - (sl_dist * 0.5)
                    if new_stop > current_stop_price: current_stop_price = new_stop
            
            else: # SHORT
                if curr_l < best_price: best_price = curr_l
                
                if not is_trailing_active and curr_l <= trailing_activation_price:
                    is_trailing_active = True
                    
                if is_trailing_active:
                    new_stop = best_price + (sl_dist * 0.5)
                    if new_stop < current_stop_price: current_stop_price = new_stop

            # 3. Stop Loss 체크 (트레일링 포함)
            if p_type == 'LONG' and curr_l <= current_stop_price:
                reason = "TrailingProfit" if is_trailing_active else "StopLoss"
                return True, current_stop_price, reason, ts
            if p_type == 'SHORT' and curr_h >= current_stop_price:
                reason = "TrailingProfit" if is_trailing_active else "StopLoss"
                return True, current_stop_price, reason, ts
        
        return False, 0, "", None

    def update_pnl_and_check_exit(self, current_close, current_high, current_low, timestamp, current_atr, precision_candles=None, sl_mult=3.0, tp_ratio=2.0):
        if not self.position: return 0
        
        entry = self.position['price']
        size = self.position['size']
        p_type = self.position['type']
        lev = self.position['leverage']
        
        funding_rate = config.FUNDING_RATE_4H / 4 if config.MAIN_TIMEFRAME == '1h' else config.FUNDING_RATE_4H
        self.balance -= (current_close * size * funding_rate)

        exit_triggered = False
        exit_price = 0
        exit_reason = ""
        exit_time = timestamp

        if precision_candles is not None and not precision_candles.empty:
            is_closed, p_price, reason, p_time = self.check_exit_with_precision(
                precision_candles, entry, size, p_type, lev, 
                current_atr * sl_mult, tp_ratio
            )
            if is_closed:
                exit_triggered = True
                exit_price = p_price
                exit_reason = reason
                exit_time = p_time
        
        # Fallback (5분봉 없을 때)
        if not exit_triggered:
            sl_dist = current_atr * sl_mult
            
            # Fallback은 보수적으로 고정 SL만 체크 (TP는 정밀 데이터에서만)
            if p_type == 'LONG':
                sl_price = entry - sl_dist
                if current_low <= sl_price:
                    exit_triggered = True; exit_price = sl_price; exit_reason = "StopLoss(Bar)"
            else:
                sl_price = entry + sl_dist
                if current_high >= sl_price:
                    exit_triggered = True; exit_price = sl_price; exit_reason = "StopLoss(Bar)"

        if exit_triggered:
            if exit_reason == "LIQUIDATION":
                logger.warning(f"!!! [LIQUIDATION] Price hit {exit_price:.2f}")
                loss = (entry * size)
                self.balance -= loss
                self.position = None
                return -loss
            else:
                self._force_close(exit_price, exit_time, exit_reason)
                return 0
        
        if p_type == 'LONG': return (current_close - entry) * size
        else: return (entry - current_close) * size

    def _force_close(self, price, timestamp, reason):
        if not self.position: return
        
        if self.position['type'] == 'LONG': real_price = price * (1 - config.SLIPPAGE)
        else: real_price = price * (1 + config.SLIPPAGE)

        size = self.position['size']
        entry = self.position['price']
        leverage = self.position['leverage']
        
        if self.position['type'] == 'LONG': pnl = (real_price - entry) * size
        else: pnl = (entry - real_price) * size
            
        fee = real_price * size * config.COMMISSION
        pnl -= fee
        
        self.balance += pnl
        
        # ROI 계산
        margin = (entry * size) / leverage
        roi = (pnl / margin) * 100 if margin > 0 else 0.0

        logger.info(f"[{timestamp}] Close {self.position['type']} ({reason}) | "
                    f"Price: {real_price:.2f} | PnL: {pnl:+.2f} ({roi:+.2f}%) | Bal: {self.balance:.0f}")
        
        self.position = None
        self.peak_price = 0

    def execute_trade(self, signal, price, qty, leverage, timestamp):
        if self.position:
            should_close = False
            if self.position['type'] == 'LONG' and signal == 'OPEN_SHORT': should_close = True
            if self.position['type'] == 'SHORT' and signal == 'OPEN_LONG': should_close = True
            if should_close: self._force_close(price, timestamp, 'SignalSwitch')

        if signal in ['OPEN_LONG', 'OPEN_SHORT'] and not self.position:
            if self.balance <= 0: return 
            
            if signal == 'OPEN_LONG':
                real_entry_price = price * (1 + config.SLIPPAGE)
                p_type = 'LONG'
            else:
                real_entry_price = price * (1 - config.SLIPPAGE)
                p_type = 'SHORT'
            
            if qty <= 0: return
            
            leverage = int(leverage)
            initial_margin = (real_entry_price * qty) / leverage
            if initial_margin > self.balance:
                qty = (self.balance * leverage) / real_entry_price

            self.balance -= (real_entry_price * qty * config.COMMISSION)
            self.position = {'type': p_type, 'price': real_entry_price, 'size': qty, 'leverage': leverage}
            self.peak_price = real_entry_price
            logger.info(f"[{timestamp}] Open {p_type} @ {real_entry_price:.2f} (Qty: {qty:.4f}, Lev: x{leverage})")