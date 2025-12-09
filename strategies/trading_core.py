import logging
import config

class TradingCore:
    def __init__(self):
        # 전역 로거 사용
        self.logger = logging.getLogger()
        self.reset()

    def reset(self):
        self.balance = config.INITIAL_BALANCE
        self.position = None  # None or {'type': 'LONG'/'SHORT', 'price': float, 'size': float, 'sl': float}
        self.history = []     # 매매 기록

    def get_unrealized_pnl(self, current_price):
        """미실현 손익금(Unrealized PnL) 계산"""
        if not self.position:
            return 0.0
        
        entry = self.position['price']
        size = self.position['size']
        
        if self.position['type'] == 'LONG':
            pnl = (current_price - entry) * size
        else: # SHORT
            pnl = (entry - current_price) * size
            
        return pnl

    def get_unrealized_pnl_pct(self, current_price):
        """미실현 수익률(%) 계산 (레버리지 미포함 순수 가격 변동분)"""
        if not self.position:
            return 0.0
        entry = self.position['price']
        pnl = self.get_unrealized_pnl(current_price)
        return (pnl / (entry * self.position['size'])) if entry else 0.0

    def _calculate_entry_size(self, price):
        """가용 자금과 레버리지를 고려한 진입 수량 계산"""
        if self.balance <= 0:
            return 0.0
            
        # 수수료 여유분(1%)을 둔 가용 자금
        usable_balance = self.balance * 0.99
        # 레버리지 적용
        leveraged_balance = usable_balance * config.LEVERAGE
        
        qty = leveraged_balance / price
        return qty

    def process_step(self, action, row_1h, timestamp, precision_candles=None):
        """
        한 스텝(1시간) 동안의 매매 로직 처리
        """
        current_close = row_1h['close']
        current_high = row_1h['high']
        current_low = row_1h['low']
        
        # [중요] 추세 필터용 지표 (EMA 200)
        # DataLoader에서 'ema_200'을 만들어준다고 가정
        ema_200 = row_1h.get('ema_200', current_close) 
        
        # ATR을 이용한 동적 SL/TP 거리 계산
        atr = row_1h.get('atr', current_close * 0.01)
        
        # 1. 포지션 보유 시 처리 (펀딩비 & 청산 체크)
        if self.position:
            # 펀딩비 차감 (약식: 1시간당 0.0025%)
            funding_cost = (self.position['price'] * self.position['size']) * (0.0001 / 4)
            self.balance -= funding_cost

            # 정밀 청산 시뮬레이션
            exit_triggered = False
            if precision_candles is not None and not precision_candles.empty:
                exit_triggered = self._check_exit_precision(precision_candles)
            else:
                exit_triggered = self._check_exit_fallback(current_high, current_low)
            
            if exit_triggered:
                return

        # 2. Action 수행 (Trend Filter 적용)
        # Action: 0=Hold, 1=Long, 2=Short, 3=Close
        
        # [Trend Filter] 추세 역행 매매 금지
        # EMA 200 위에 있으면(상승세) -> Short 금지
        # EMA 200 아래에 있으면(하락세) -> Long 금지
        is_uptrend = current_close > ema_200
        
        final_action = action
        if action == 1 and not is_uptrend: # 하락장에서 롱 시도
            final_action = 0 # 무시 (Hold)
        elif action == 2 and is_uptrend:   # 상승장에서 숏 시도
            final_action = 0 # 무시 (Hold)

        # 2-1. Close Action
        if final_action == 3 and self.position:
            self._close_position(current_close, reason="Signal Close")
            
        # 2-2. Open Long
        elif final_action == 1:
            if self.position:
                if self.position['type'] == 'SHORT':
                    self._close_position(current_close, reason="Switch Long")
                    self._open_position('LONG', current_close, atr)
            else:
                self._open_position('LONG', current_close, atr)
                
        # 2-3. Open Short
        elif final_action == 2:
            if self.position:
                if self.position['type'] == 'LONG':
                    self._close_position(current_close, reason="Switch Short")
                    self._open_position('SHORT', current_close, atr)
            else:
                self._open_position('SHORT', current_close, atr)

    def _open_position(self, p_type, price, atr):
        """포지션 진입 로직"""
        real_price = price * (1 + config.SLIPPAGE) if p_type == 'LONG' else price * (1 - config.SLIPPAGE)
        
        size = self._calculate_entry_size(real_price)
        if size <= 0: return

        fee = (real_price * size) * config.FEE_RATE
        self.balance -= fee
        
        # 초기 SL 설정 (ATR * 2)
        sl_dist = atr * 2.0
        if p_type == 'LONG':
            sl_price = real_price - sl_dist
        else:
            sl_price = real_price + sl_dist

        # [로그 강화] 진입 기록
        self.logger.info(f"   [TRADE] OPEN {p_type:<5} @ {real_price:.2f} | Size: {size:.4f} | SL: {sl_price:.2f}")

        self.position = {
            'type': p_type,
            'price': real_price,
            'size': size,
            'sl': sl_price,
            'base_sl_dist': sl_dist, 
            'highest_price': real_price if p_type == 'LONG' else 0, 
            'lowest_price': real_price if p_type == 'SHORT' else float('inf') 
        }

    def _close_position(self, price, reason="Close"):
        """포지션 종료 로직"""
        if not self.position: return

        real_price = price * (1 - config.SLIPPAGE) if self.position['type'] == 'LONG' else price * (1 + config.SLIPPAGE)
        
        entry = self.position['price']
        size = self.position['size']
        
        if self.position['type'] == 'LONG':
            pnl = (real_price - entry) * size
        else:
            pnl = (entry - real_price) * size
            
        fee = (real_price * size) * config.FEE_RATE
        self.balance += (pnl - fee)
        
        # [로그 강화] 청산 기록 (ROI 포함)
        roi = (pnl / (entry * size)) * 100
        self.logger.info(f"   [TRADE] CLOSE {self.position['type']:<5} @ {real_price:.2f} | PnL: ${pnl:+.2f} ({roi:+.2f}%) | Why: {reason}")
        
        self.position = None

    def _check_exit_precision(self, candles_5m):
        """5분봉 데이터를 순회하며 SL/TP/Trailing Stop 체크"""
        if not self.position: return False
        
        p_type = self.position['type']
        entry_price = self.position['price']
        
        # 청산가 계산
        liq_threshold = 1.0 / config.LEVERAGE
        if p_type == 'LONG':
            liq_price = entry_price * (1 - liq_threshold + 0.005) 
        else:
            liq_price = entry_price * (1 + liq_threshold - 0.005)

        for _, row in candles_5m.iterrows():
            curr_high = row['high']
            curr_low = row['low']
            
            # 1. 강제 청산(Liquidation)
            if p_type == 'LONG' and curr_low <= liq_price:
                self._close_position(liq_price, reason="LIQUIDATION")
                return True
            if p_type == 'SHORT' and curr_high >= liq_price:
                self._close_position(liq_price, reason="LIQUIDATION")
                return True

            # 2. Stop Loss
            if p_type == 'LONG' and curr_low <= self.position['sl']:
                self._close_position(self.position['sl'], reason="StopLoss")
                return True
            if p_type == 'SHORT' and curr_high >= self.position['sl']:
                self._close_position(self.position['sl'], reason="StopLoss")
                return True
            
            # 3. Trailing Stop & Break Even
            self._update_stops(curr_high, curr_low, entry_price)

        return False

    def _check_exit_fallback(self, high, low):
        """정밀 데이터 없을 때 1시간봉 기준 체크"""
        if not self.position: return False
        
        if self.position['type'] == 'LONG':
            if low <= self.position['sl']:
                self._close_position(self.position['sl'], reason="StopLoss(Fallback)")
                return True
        else:
            if high >= self.position['sl']:
                self._close_position(self.position['sl'], reason="StopLoss(Fallback)")
                return True
        return False

    def _update_stops(self, curr_high, curr_low, entry_price):
        """Trailing Stop 및 Break Even 로직"""
        dist = self.position['base_sl_dist']
        
        if self.position['type'] == 'LONG':
            if curr_high > self.position['highest_price']:
                self.position['highest_price'] = curr_high
                
                # Break Even (수익이 꽤 났을 때 본절로 이동)
                if curr_high > entry_price + (dist * 1.5):
                    new_sl = max(self.position['sl'], entry_price * 1.002) 
                    self.position['sl'] = new_sl
                
                # Trailing Stop (더 큰 수익 시 추격)
                if curr_high > entry_price + (dist * 3.0):
                    new_sl = max(self.position['sl'], curr_high - dist)
                    self.position['sl'] = new_sl
                    
        else: # SHORT
            if curr_low < self.position['lowest_price']:
                self.position['lowest_price'] = curr_low
                
                if curr_low < entry_price - (dist * 1.5):
                    new_sl = min(self.position['sl'], entry_price * 0.998)
                    self.position['sl'] = new_sl
                    
                if curr_low < entry_price - (dist * 3.0):
                    new_sl = min(self.position['sl'], curr_low + dist)
                    self.position['sl'] = new_sl