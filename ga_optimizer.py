# ga_optimizer.py
import numpy as np
import random
import config
from numba import njit
from joblib import Parallel, delayed

# ---------------------------------------------------------
# [핵심] Numba JIT 컴파일된 백테스트 코어 (본절 로직 탑재)
# ---------------------------------------------------------
@njit(fastmath=True, nogil=True)
def fast_backtest_core(
    opens, highs, lows, closes, atrs, scores,
    balance,
    entry_threshold, sl_mul, risk_scale, tp_ratio,
    min_lev, max_lev,
    fee_rate, enable_short,
    be_trigger_pct # [추가] 본절 트리거 비율
):
    # 상태 변수
    position_type = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    position_size = 0.0
    sl_price = 0.0
    
    # Trailing & BE 상태 관리
    tp_activation_price = 0.0
    is_trailing = False
    
    be_activation_price = 0.0
    is_be_active = False
    
    best_price = 0.0
    
    init_balance = balance
    peak_balance = balance
    max_drawdown = 0.0
    trade_count = 0
    win_count = 0
    
    n = len(closes)
    
    for i in range(n - 1):
        if balance <= 0: break
        
        # MDD 갱신
        if balance > peak_balance: peak_balance = balance
        dd = (peak_balance - balance) / peak_balance
        if dd > max_drawdown: max_drawdown = dd
        
        score = scores[i]
        confidence = abs(score)
        
        next_o = opens[i+1]
        next_h = highs[i+1]
        next_l = lows[i+1]
        current_atr = atrs[i]
        
        # --- 청산 및 관리 로직 ---
        if position_type != 0:
            closed = False
            exit_price = 0.0
            
            # 1. Long Position 관리
            if position_type == 1:
                if next_h > best_price: best_price = next_h
                
                # A. 본절(Break-Even) 체크
                if not is_be_active and next_h >= be_activation_price:
                    is_be_active = True
                    # 진입가 + 0.2% (수수료 커버) 위로 손절 이동
                    new_sl = entry_price * 1.002
                    if new_sl > sl_price: sl_price = new_sl
                
                # B. 트레일링 스탑 체크
                if not is_trailing and next_h >= tp_activation_price:
                    is_trailing = True
                
                if is_trailing:
                    sl_dist = current_atr * sl_mul
                    # 최고가 대비 0.5배 거리로 추격
                    new_sl = best_price - (sl_dist * 0.5)
                    if new_sl > sl_price: sl_price = new_sl
                    
                # C. 손절/익절 실행 (업데이트된 SL 기준)
                if next_l <= sl_price:
                    exit_price = sl_price; closed = True
            
            # 2. Short Position 관리
            else: 
                if next_l < best_price: best_price = next_l
                
                # A. 본절(Break-Even) 체크
                if not is_be_active and next_l <= be_activation_price:
                    is_be_active = True
                    # 진입가 - 0.2% (수수료 커버) 아래로 손절 이동
                    new_sl = entry_price * 0.998
                    if new_sl < sl_price: sl_price = new_sl
                    
                # B. 트레일링 스탑 체크
                if not is_trailing and next_l <= tp_activation_price:
                    is_trailing = True
                    
                if is_trailing:
                    sl_dist = current_atr * sl_mul
                    # 최저가 대비 0.5배 거리로 추격
                    new_sl = best_price + (sl_dist * 0.5)
                    if new_sl < sl_price: sl_price = new_sl
                    
                # C. 손절/익절 실행
                if next_h >= sl_price:
                    exit_price = sl_price; closed = True
            
            # 3. 포지션 종료 처리
            if closed:
                pnl = 0.0
                if position_type == 1: pnl = (exit_price - entry_price) * position_size
                else: pnl = (entry_price - exit_price) * position_size
                
                # 수수료 차감
                cost = exit_price * position_size * fee_rate
                balance += (pnl - cost)
                
                if pnl > 0: win_count += 1
                trade_count += 1
                position_type = 0
                
                if balance > peak_balance: peak_balance = balance
                dd = (peak_balance - balance) / peak_balance
                if dd > max_drawdown: max_drawdown = dd

        # --- 진입 로직 ---
        if position_type == 0 and confidence > entry_threshold:
            signal = 1 if score > 0 else -1
            if signal == -1 and not enable_short: continue
                
            risk_amt = balance * risk_scale
            sl_dist = current_atr * sl_mul
            
            if sl_dist > 0:
                qty_risk = risk_amt / sl_dist
                
                ratio = (confidence - entry_threshold) / (1.0 - entry_threshold + 1e-9)
                ratio = min(max(ratio, 0.0), 1.0)
                target_lev = min_lev + (ratio * (max_lev - min_lev))
                
                max_qty = (balance * target_lev) / next_o
                qty = min(qty_risk, max_qty)
                
                if qty > 0:
                    real_entry = next_o * (1 + fee_rate) if signal == 1 else next_o * (1 - fee_rate)
                    entry_cost = real_entry * qty * fee_rate
                    
                    if balance > entry_cost:
                        balance -= entry_cost
                        position_type = signal
                        entry_price = real_entry
                        position_size = qty
                        best_price = real_entry
                        
                        # 초기 상태 설정
                        is_trailing = False
                        is_be_active = False
                        
                        if signal == 1:
                            sl_price = real_entry - sl_dist
                            # TP 비율은 트레일링 시작점으로 사용
                            tp_activation_price = real_entry + (sl_dist * tp_ratio)
                            be_activation_price = real_entry * (1 + be_trigger_pct)
                        else:
                            sl_price = real_entry + sl_dist
                            tp_activation_price = real_entry - (sl_dist * tp_ratio)
                            be_activation_price = real_entry * (1 - be_trigger_pct)

    return balance, trade_count, max_drawdown, win_count

class GeneticOptimizer:
    def __init__(self):
        self.settings = config.GA_SETTINGS
        self.gene_ranges = config.GA_GENE_RANGES

    def create_individual(self):
        return {k: random.uniform(v[0], v[1]) for k, v in self.gene_ranges.items()}

    def mutate(self, individual):
        for k in individual:
            if random.random() < self.settings['mutation_rate']:
                low, high = self.gene_ranges[k]
                individual[k] = random.uniform(low, high)
        return individual

    def crossover(self, p1, p2):
        child = {}
        for k in p1:
            child[k] = p1[k] if random.random() > 0.5 else p2[k]
        return child

    def optimize(self, df, scores):
        # 데이터 준비
        opens = df['open'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        
        # 원본 ATR 사용
        if 'atr_origin' in df: atrs = df['atr_origin'].values.astype(np.float64)
        else: atrs = df['atr'].values.astype(np.float64)
            
        scores = scores.astype(np.float64)
        
        fee_rate = config.COMMISSION + config.SLIPPAGE
        min_lev = float(config.MIN_LEVERAGE)
        max_lev = float(config.MAX_LEVERAGE)
        enable_short = config.ENABLE_SHORT
        init_bal = float(config.INITIAL_BALANCE)
        
        # [수정] 본절 트리거 비율 가져오기
        be_trigger_pct = float(config.BE_TRIGGER_PCT)

        pop_size = self.settings['population_size']
        population = [self.create_individual() for _ in range(pop_size)]
        
        for gen in range(self.settings['generations']):
            results = Parallel(n_jobs=-1)(
                delayed(fast_backtest_core)(
                    opens, highs, lows, closes, atrs, scores,
                    init_bal,
                    ind['entry_threshold'], ind['sl_mul'], ind['risk_scale'], ind['tp_ratio'],
                    min_lev, max_lev, fee_rate, enable_short,
                    be_trigger_pct # 인자 전달
                ) for ind in population
            )
            
            fitness_scores = []
            for (bal, cnt, mdd, wins), ind in zip(results, population):
                roi = (bal - init_bal) / init_bal
                
                # 최소 거래 횟수 충족 및 파산 방지
                if cnt < 3: fitness = -1.0
                elif bal < init_bal * 0.8: fitness = -10.0
                else:
                    # Risk-Adjusted Return (Calmar-like)
                    fitness = roi / (mdd + 0.05)
                
                fitness_scores.append((fitness, ind))
            
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            next_gen = [x[1] for x in fitness_scores[:self.settings['elitism']]]
            
            while len(next_gen) < pop_size:
                p1 = random.choice(fitness_scores[:int(pop_size/2)])[1]
                p2 = random.choice(fitness_scores[:int(pop_size/2)])[1]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
                
            population = next_gen
            
        return fitness_scores[0][1]