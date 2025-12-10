import logging
import config
import pandas as pd

class TradingCore:
    def __init__(self):
        self.logger = logging.getLogger()
        self.rules = config.TRADING_RULES
        self.reset()

    def reset(self):
        self.balance = config.INITIAL_BALANCE
        self.position = None
        self.history = []        # 매매 이력 저장용 (분석용)
        self.trade_count = 0

    def get_unrealized_pnl(self, current_price):
        if not self.position: return 0.0
        entry = self.position['price']
        size = self.position['size']
        if self.position['type'] == 'LONG':
            return (current_price - entry) * size
        else:
            return (entry - current_price) * size

    def get_unrealized_pnl_pct(self, current_price):
        if not self.position: return 0.0
        entry = self.position['price']
        pnl = self.get_unrealized_pnl(current_price)
        return (pnl / (entry * self.position['size'])) if entry else 0.0

    def _calculate_risk_based_size(self, price, sl_price):
        if self.balance <= 0: return 0.0
        
        risk_amount = self.balance * self.rules['risk_per_trade']
        loss_per_unit = abs(price - sl_price)
        if loss_per_unit == 0: return 0.0
        
        qty = risk_amount / loss_per_unit
        max_qty = (self.balance * config.MAX_LEVERAGE) / price
        qty = min(qty, max_qty)
        return qty

    def process_step(self, action, row_1h, timestamp, precision_candles=None):
        """
        timestamp: 현재 백테스트 시점의 시간 (datetime)
        """
        current_close = row_1h['close']
        current_high = row_1h['high']
        current_low = row_1h['low']
        
        ema_trend_4h = row_1h.get('ema_trend_4h', 0)
        atr = row_1h.get('atr', current_close * 0.01)
        
        # 1. 포지션 관리
        if self.position:
            funding_cost = (self.position['price'] * self.position['size']) * self.rules['funding_rate_hourly']
            self.balance -= funding_cost

            exit_triggered = False
            if precision_candles is not None and not precision_candles.empty:
                exit_triggered = self._check_exit_precision(precision_candles, timestamp)
            else:
                exit_triggered = self._check_exit_fallback(current_high, current_low, timestamp)
            
            if exit_triggered: return

        # 2. 추세 필터
        trend_up = (current_close > ema_trend_4h) if ema_trend_4h > 0 else True
        trend_down = (current_close < ema_trend_4h) if ema_trend_4h > 0 else True

        # Action Logic
        if action == 3 and self.position:
            self._close_position(current_close, timestamp, reason="Signal Close")
            
        elif action == 1:
            if self.position:
                if self.position['type'] == 'SHORT':
                    self._close_position(current_close, timestamp, reason="Switch Long")
                    if trend_up: self._open_position('LONG', current_close, atr, timestamp)
            elif trend_up:
                self._open_position('LONG', current_close, atr, timestamp)
                
        elif action == 2:
            if self.position:
                if self.position['type'] == 'LONG':
                    self._close_position(current_close, timestamp, reason="Switch Short")
                    if trend_down: self._open_position('SHORT', current_close, atr, timestamp)
            elif trend_down:
                self._open_position('SHORT', current_close, atr, timestamp)

    def _open_position(self, p_type, price, atr, timestamp):
        real_price = price * (1 + config.SLIPPAGE) if p_type == 'LONG' else price * (1 - config.SLIPPAGE)
        
        sl_dist = atr * self.rules['sl_atr_multiplier']
        sl_price = real_price - sl_dist if p_type == 'LONG' else real_price + sl_dist
        
        size = self._calculate_risk_based_size(real_price, sl_price)
        if (size * real_price) < self.rules['min_trade_amount']:
            return

        fee = (real_price * size) * config.FEE_RATE
        self.balance -= fee
        
        leverage_used = (size * real_price) / self.balance
        
        # [수정] 백테스트 시간 표시
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
        self.logger.info(f"[{ts_str}] [TRADE] OPEN {p_type:<5} @ {real_price:.2f} | Size: {size:.4f} ({leverage_used:.1f}x) | SL: {sl_price:.2f}")

        self.position = {
            'type': p_type,
            'entry_price': real_price, # 명확한 이름 사용
            'price': real_price,       # 호환성 유지
            'size': size,
            'sl': sl_price,
            'base_sl_dist': sl_dist, 
            'highest_price': real_price if p_type == 'LONG' else 0, 
            'lowest_price': real_price if p_type == 'SHORT' else float('inf'),
            'open_time': timestamp
        }

    def _close_position(self, price, timestamp, reason="Close"):
        if not self.position: return

        real_price = price * (1 - config.SLIPPAGE) if self.position['type'] == 'LONG' else price * (1 + config.SLIPPAGE)
        entry = self.position['entry_price']
        size = self.position['size']
        
        if self.position['type'] == 'LONG': pnl = (real_price - entry) * size
        else: pnl = (entry - real_price) * size
            
        fee = (real_price * size) * config.FEE_RATE
        self.balance += (pnl - fee)
        
        roi = (pnl / (entry * size)) * 100
        ts_str = timestamp.strftime('%Y-%m-%d %H:%M')
        
        self.logger.info(f"[{ts_str}] [TRADE] CLOSE {self.position['type']:<5} @ {real_price:.2f} | PnL: ${pnl:+.2f} ({roi:+.2f}%) | Why: {reason}")
        
        # [추가] 거래 이력 저장
        self.history.append({
            'Open Time': self.position['open_time'],
            'Close Time': timestamp,
            'Type': self.position['type'],
            'Entry Price': entry,
            'Exit Price': real_price,
            'Size': size,
            'PnL': pnl,
            'ROI(%)': roi,
            'Reason': reason,
            'Balance': self.balance
        })
        self.trade_count += 1
        self.position = None

    def _check_exit_precision(self, candles_5m, timestamp_1h):
        if not self.position: return False
        
        p_type = self.position['type']
        
        for ts_5m, row in candles_5m.iterrows():
            curr_high, curr_low = row['high'], row['low']
            
            # SL Check
            if (p_type == 'LONG' and curr_low <= self.position['sl']) or (p_type == 'SHORT' and curr_high >= self.position['sl']):
                self._close_position(self.position['sl'], ts_5m, "StopLoss")
                return True
                
            # Trailing Stop Update
            self._update_stops(curr_high, curr_low, self.position['entry_price'])
            
        return False

    def _check_exit_fallback(self, high, low, timestamp):
        if not self.position: return False
        if self.position['type'] == 'LONG':
            if low <= self.position['sl']:
                self._close_position(self.position['sl'], timestamp, "StopLoss(Fallback)")
                return True
        else:
            if high >= self.position['sl']:
                self._close_position(self.position['sl'], timestamp, "StopLoss(Fallback)")
                return True
        return False

    def _update_stops(self, curr_high, curr_low, entry_price):
        dist = self.position['base_sl_dist']
        atr = dist / self.rules['sl_atr_multiplier']
        
        trigger_atr = self.rules['tp_trigger_atr']
        gap_atr = self.rules['trailing_gap_atr']
        
        if self.position['type'] == 'LONG':
            if curr_high > self.position['highest_price']:
                self.position['highest_price'] = curr_high
                # 본절
                if curr_high > entry_price + (atr * trigger_atr):
                    self.position['sl'] = max(self.position['sl'], entry_price * 1.001)
                # 트레일링
                if curr_high > entry_price + (atr * (trigger_atr + 1.0)):
                    self.position['sl'] = max(self.position['sl'], curr_high - (atr * gap_atr))
                    
        else: # SHORT
            if curr_low < self.position['lowest_price']:
                self.position['lowest_price'] = curr_low
                if curr_low < entry_price - (atr * trigger_atr):
                    self.position['sl'] = min(self.position['sl'], entry_price * 0.999)
                if curr_low < entry_price - (atr * (trigger_atr + 1.0)):
                    self.position['sl'] = min(self.position['sl'], curr_low + (atr * gap_atr))