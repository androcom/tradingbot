# ga_optimizer.py
import random
import numpy as np
import copy
import logging
import config

logger = logging.getLogger("BacktestLogger")

class GeneticOptimizer:
    def __init__(self):
        self.bounds = config.GENE_RANGES

    def create_individual(self):
        """랜덤한 유전자(파라미터 세트) 생성"""
        return {
            'base_threshold': random.uniform(*self.bounds['base_threshold']),
            'vol_impact': random.uniform(*self.bounds['vol_impact']),
            'sl_mul': random.uniform(*self.bounds['sl_mul']),
            'tp_ratio': random.uniform(*self.bounds['tp_ratio'])
        }

    def crossover(self, parent1, parent2):
        """두 부모 유전자를 섞어 자식 생성 (Uniform Crossover)"""
        child = {}
        for key in parent1:
            if random.random() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(self, individual):
        """돌연변이: 일정 확률로 유전자 값을 랜덤 변경"""
        for key in individual:
            if random.random() < config.GA_MUTATION_RATE:
                individual[key] = random.uniform(*self.bounds[key])
        return individual

    def fast_backtest(self, ohlcv_data, predictions, genes):
        """
        초고속 백테스트 (GA용)
        - 복잡한 로깅이나 객체 생성 없이 수치 계산만 수행
        - ohlcv_data: [(open, high, low, close), ...] numpy array
        - predictions: [(score, pred_vol), ...] numpy array
        """
        balance = config.INITIAL_BALANCE
        position = None # (type, entry_price, size, leverage, sl, tp)
        
        # 데이터 언패킹 (속도 최적화)
        opens = ohlcv_data[:, 0]
        highs = ohlcv_data[:, 1]
        lows = ohlcv_data[:, 2]
        closes = ohlcv_data[:, 3]
        
        scores = predictions[:, 0]
        pred_vols = predictions[:, 1]
        
        n = len(closes)
        
        for i in range(n - 1):
            if balance <= 0: break
            
            # 1. 포지션 관리 (청산)
            if position is not None:
                p_type, entry, size, lev, sl, tp = position
                
                # 강제청산 체크
                liq_rate = 1.0 / lev
                liq_price = entry * (1 - liq_rate) if p_type == 1 else entry * (1 + liq_rate)
                
                is_liq = False
                if p_type == 1: # LONG
                    if lows[i+1] <= liq_price: is_liq = True
                else: # SHORT
                    if highs[i+1] >= liq_price: is_liq = True
                
                if is_liq:
                    loss = (entry * size) / lev
                    balance -= loss
                    position = None
                    continue

                # SL/TP 체크
                close_signal = False
                exec_price = closes[i+1]
                
                if p_type == 1: # LONG
                    if lows[i+1] <= sl:
                        close_signal = True; exec_price = sl
                    elif highs[i+1] >= tp:
                        close_signal = True; exec_price = tp
                else: # SHORT
                    if highs[i+1] >= sl:
                        close_signal = True; exec_price = sl
                    elif lows[i+1] <= tp:
                        close_signal = True; exec_price = tp
                
                if close_signal:
                    # 수수료/슬리피지 단순화 적용
                    if p_type == 1: pnl = (exec_price - entry) * size
                    else: pnl = (entry - exec_price) * size
                    balance += pnl
                    balance -= (exec_price * size * config.COMMISSION)
                    position = None
                    
            # 2. 신규 진입 (포지션 없을 때만)
            if position is None:
                score = scores[i]
                vol = pred_vols[i]
                curr_price = opens[i+1] # 다음 봉 시가 진입
                
                # [동적 파라미터 적용]
                threshold = genes['base_threshold'] + (vol * genes['vol_impact'])
                threshold = min(threshold, 0.9)
                
                confidence = abs(score)
                if confidence < threshold: continue
                
                # 신호 결정
                signal = 0 # 0: None, 1: Long, -1: Short
                if score > 0: signal = 1
                elif score < 0 and config.ENABLE_SHORT: signal = -1
                
                if signal == 0: continue
                
                # 레버리지 및 수량 계산
                raw_lev = 1 + (confidence * (config.MAX_LEVERAGE - 1)) / 0.8
                lev = int(np.clip(round(raw_lev), config.MIN_LEVERAGE, config.MAX_LEVERAGE))
                
                sl_dist = curr_price * (max(vol, 0.005) * genes['sl_mul'])
                tp_dist = sl_dist * genes['tp_ratio']
                
                risk_amt = balance * config.MAX_RISK_PER_TRADE_CAP
                qty = risk_amt / sl_dist if sl_dist > 0 else 0
                max_qty = (balance * lev) / curr_price
                qty = min(qty, max_qty)
                
                if qty <= 0: continue
                
                # 포지션 생성
                if signal == 1:
                    sl_price = curr_price - sl_dist
                    tp_price = curr_price + tp_dist
                else:
                    sl_price = curr_price + sl_dist
                    tp_price = curr_price - tp_dist
                
                balance -= (curr_price * qty * config.COMMISSION)
                position = (signal, curr_price, qty, lev, sl_price, tp_price)

        return balance

    def optimize(self, ohlcv_data, predictions):
        """유전 알고리즘 메인 루프"""
        # 초기 개체군 생성
        population = [self.create_individual() for _ in range(config.GA_POPULATION_SIZE)]
        best_gene = None
        best_fitness = -float('inf')

        for gen in range(config.GA_GENERATIONS):
            # 적합도 평가 (병렬 처리 가능하나 여기선 단순 루프)
            fitness_scores = []
            for ind in population:
                final_bal = self.fast_backtest(ohlcv_data, predictions, ind)
                # 수익률이 0 이하거나 파산하면 페널티
                fitness = final_bal if final_bal > 0 else -999999
                fitness_scores.append((fitness, ind))
            
            # 정렬 (내림차순)
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 최고 기록 갱신
            if fitness_scores[0][0] > best_fitness:
                best_fitness = fitness_scores[0][0]
                best_gene = copy.deepcopy(fitness_scores[0][1])
            
            # 엘리트 선택
            next_generation = [x[1] for x in fitness_scores[:config.GA_ELITISM]]
            
            # 나머지 채우기 (토너먼트 선택 + 교배 + 변이)
            while len(next_generation) < config.GA_POPULATION_SIZE:
                # 토너먼트 선택
                candidates = random.sample(fitness_scores, 5)
                parent1 = max(candidates, key=lambda x: x[0])[1]
                candidates = random.sample(fitness_scores, 5)
                parent2 = max(candidates, key=lambda x: x[0])[1]
                
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)
            
            population = next_generation

        return best_gene, best_fitness