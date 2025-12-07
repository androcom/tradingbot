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
        [수정] 본절(BE) + 트레일링 스탑 복합 로직
        """
        if candles_5m.empty: return False, 0, "", None

        liq_threshold = 1.0 / float(lev)
        if p_type == 'LONG':
            liq_price = entry * (1 - liq_threshold + 0.005) 
            current_sl_price = entry - sl_dist
            tp_activation_price = entry + (sl_dist * tp_ratio)
            be_activation_price = entry * (1 + config.BE_TRIGGER_PCT)
        else:
            liq_price = entry * (1 + liq_threshold - 0.005)
            current_sl_price = entry + sl_dist
            tp_activation_price = entry - (sl_dist * tp_ratio)
            be_activation_price = entry * (1 - config.BE_TRIGGER_PCT)
            
        is_trailing = False
        is_be_active = False
        best_price = entry 
        
        for ts, row in candles_5m.iterrows():
            curr_h = row['high']
            curr_l = row['low']
            
            # 1. 청산 체크
            if p_type == 'LONG' and curr_l <= liq_price: return True, liq_price, "LIQUIDATION", ts
            if p_type == 'SHORT' and curr_h >= liq_price: return True, liq_price, "LIQUIDATION", ts

            # 2. 로직 업데이트
            if p_type == 'LONG':
                if curr_h > best_price: best_price = curr_h
                
                # 본절 체크
                if not is_be_active and curr_h >= be_activation_price:
                    is_be_active = True
                    if (entry * 1.002) > current_sl_price:
                        current_sl_price = entry * 1.002
                
                # 트레일링 체크
                if not is_trailing and curr_h >= tp_activation_price:
                    is_trailing = True
                
                if is_trailing:
                    new_sl = best_price - (sl_dist * 0.5)
                    if new_sl > current_sl_price: current_sl_price = new_sl

            else: # SHORT
                if curr_l < best_price: best_price = curr_l
                
                if not is_be_active and curr_l <= be_activation_price:
                    is_be_active = True
                    if (entry * 0.998) < current_sl_price:
                        current_sl_price = entry * 0.998
                
                if not is_trailing and curr_l <= tp_activation_price:
                    is_trailing = True
                
                if is_trailing:
                    new_sl = best_price + (sl_dist * 0.5)
                    if new_sl < current_sl_price: current_sl_price = new_sl

            # 3. SL 실행
            if p_type == 'LONG' and curr_l <= current_sl_price:
                reason = "TrailingProfit" if is_trailing else ("BreakEven" if is_be_active else "StopLoss")
                return True, current_sl_price, reason, ts
            if p_type == 'SHORT' and curr_h >= current_sl_price:
                reason = "TrailingProfit" if is_trailing else ("BreakEven" if is_be_active else "StopLoss")
                return True, current_sl_price, reason, ts
        
        return False, 0, "", None

    def update_pnl_and_check_exit(self, current_close, current_high, current_low, timestamp, current_atr, precision_candles=None, sl_mult=3.0, tp_ratio=2.0):
        if not self.position: return 0
        
        entry = self.position['price']
        size = self.position['size']
        p_type = self.position['type']
        
        funding_rate = config.FUNDING_RATE_4H / 4 if config.MAIN_TIMEFRAME == '1h' else config.FUNDING_RATE_4H
        self.balance -= (current_close * size * funding_rate)

        exit_triggered = False
        exit_price = 0
        exit_reason = ""
        exit_time = timestamp

        if precision_candles is not None and not precision_candles.empty:
            is_closed, p_price, reason, p_time = self.check_exit_with_precision(
                precision_candles, entry, size, p_type, self.position['leverage'], 
                current_atr * sl_mult, tp_ratio
            )
            if is_closed:
                exit_triggered = True; exit_price = p_price; exit_reason = reason; exit_time = p_time
        
        if not exit_triggered:
            sl_dist = current_atr * sl_mult
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