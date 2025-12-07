# ga_optimizer.py
import numpy as np
import random
import config
from numba import njit
from joblib import Parallel, delayed

# ---------------------------------------------------------
# [핵심] Numba JIT 컴파일된 백테스트 코어 (MDD 계산 추가)
# ---------------------------------------------------------
@njit(fastmath=True, nogil=True)
def fast_backtest_core(
    opens, highs, lows, closes, atrs, scores,
    balance,
    entry_threshold, sl_mul, risk_scale, tp_ratio,
    min_lev, max_lev,
    fee_rate, enable_short
):
    # 상태 변수
    position_type = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    position_size = 0.0
    sl_price = 0.0
    tp_price = 0.0
    
    init_balance = balance
    peak_balance = balance
    max_drawdown = 0.0
    trade_count = 0
    win_count = 0
    
    n = len(closes)
    
    for i in range(n - 1):
        if balance <= 0: break
        
        # MDD 갱신 (매 봉마다)
        if balance > peak_balance: peak_balance = balance
        dd = (peak_balance - balance) / peak_balance
        if dd > max_drawdown: max_drawdown = dd
        
        score = scores[i]
        confidence = abs(score)
        
        next_o = opens[i+1]
        next_h = highs[i+1]
        next_l = lows[i+1]
        current_atr = atrs[i]
        
        # --- 청산 로직 ---
        if position_type != 0:
            closed = False
            exit_price = 0.0
            pnl = 0.0
            
            # Long 청산 체크
            if position_type == 1:
                if next_l <= sl_price:
                    exit_price = sl_price; closed = True
                elif next_h >= tp_price:
                    exit_price = tp_price; closed = True
            # Short 청산 체크
            else:
                if next_h >= sl_price:
                    exit_price = sl_price; closed = True
                elif next_l <= tp_price:
                    exit_price = tp_price; closed = True
            
            if closed:
                if position_type == 1: pnl = (exit_price - entry_price) * position_size
                else: pnl = (entry_price - exit_price) * position_size
                
                # 수수료 차감
                cost = exit_price * position_size * fee_rate
                balance += (pnl - cost)
                
                if pnl > 0: win_count += 1
                trade_count += 1
                position_type = 0
                
                # 청산 후 MDD 다시 체크
                if balance > peak_balance: peak_balance = balance
                dd = (peak_balance - balance) / peak_balance
                if dd > max_drawdown: max_drawdown = dd

        # --- 진입 로직 ---
        # 포지션이 없고, AI 신호가 강할 때
        if position_type == 0 and confidence > entry_threshold:
            signal = 1 if score > 0 else -1
            if signal == -1 and not enable_short: continue
                
            risk_amt = balance * risk_scale
            sl_dist = current_atr * sl_mul
            
            if sl_dist > 0:
                qty_risk = risk_amt / sl_dist
                
                # 레버리지 계산
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
                        
                        if signal == 1:
                            sl_price = real_entry - sl_dist
                            tp_price = real_entry + (sl_dist * tp_ratio)
                        else:
                            sl_price = real_entry + sl_dist
                            tp_price = real_entry - (sl_dist * tp_ratio)

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
        
        # 원본 ATR 사용 (DataProcessor 수정 필요 없이 여기서 처리 가능하면 좋음)
        if 'atr_origin' in df: atrs = df['atr_origin'].values.astype(np.float64)
        else: atrs = df['atr'].values.astype(np.float64)
            
        scores = scores.astype(np.float64)
        
        fee_rate = config.COMMISSION + config.SLIPPAGE
        min_lev = float(config.MIN_LEVERAGE)
        max_lev = float(config.MAX_LEVERAGE)
        enable_short = config.ENABLE_SHORT
        init_bal = float(config.INITIAL_BALANCE)

        pop_size = self.settings['population_size']
        population = [self.create_individual() for _ in range(pop_size)]
        
        for gen in range(self.settings['generations']):
            results = Parallel(n_jobs=-1)(
                delayed(fast_backtest_core)(
                    opens, highs, lows, closes, atrs, scores,
                    init_bal,
                    ind['entry_threshold'], ind['sl_mul'], ind['risk_scale'], ind['tp_ratio'],
                    min_lev, max_lev, fee_rate, enable_short
                ) for ind in population
            )
            
            fitness_scores = []
            for (bal, cnt, mdd, wins), ind in zip(results, population):
                # [수정] Fitness Function: Risk-Adjusted Return
                # 수익률이 높아도 MDD가 크면 점수 대폭 삭감
                roi = (bal - init_bal) / init_bal
                
                # 페널티 조건
                if cnt < 3: # 거래가 너무 적으면 무효
                    fitness = -1.0
                elif bal < init_bal * 0.8: # 원금 20% 이상 손실시 탈락
                    fitness = -10.0
                else:
                    # Calmar Ratio 유사 지표: ROI / (MDD + 0.1)
                    # MDD가 0일 경우를 대비해 0.05~0.1 정도 더해줌
                    fitness = roi / (mdd + 0.05)
                
                fitness_scores.append((fitness, ind))
            
            # 정렬 (내림차순)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 엘리트 선택
            next_gen = [x[1] for x in fitness_scores[:self.settings['elitism']]]
            
            # 다음 세대 생성
            while len(next_gen) < pop_size:
                p1 = random.choice(fitness_scores[:int(pop_size/2)])[1]
                p2 = random.choice(fitness_scores[:int(pop_size/2)])[1]
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_gen.append(child)
                
            population = next_gen
            
        # 최적해 반환
        return fitness_scores[0][1]