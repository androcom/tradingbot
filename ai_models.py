# ai_models.py
import numpy as np
import os
import joblib
import logging
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import config

logger = logging.getLogger("BacktestLogger")

# GPU 메모리 할당 문제 방지
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: logger.error(e)

class HybridEnsemble:
    def __init__(self, symbol):
        self.symbol = symbol
        safe_symbol = symbol.replace('/', '_')
        
        # 모델 저장 경로 설정
        self.xgb_path = os.path.join(config.MODEL_DIR, f'xgb_{safe_symbol}.pkl')
        self.lstm_path = os.path.join(config.MODEL_DIR, f'lstm_{safe_symbol}.h5')

        self.xgb_model = None
        self.lstm_model = None

    # [복원] 모델 저장 메서드 추가
    def save_models(self):
        try:
            if self.xgb_model:
                joblib.dump(self.xgb_model, self.xgb_path)
            
            if self.lstm_model:
                self.lstm_model.save(self.lstm_path)
                
            # logger.info(f"   [Model] Saved to {config.MODEL_DIR}")
        except Exception as e:
            logger.error(f"   [Model] Save failed: {e}")

    def build_lstm(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # 과적합 방지를 위해 모델 복잡도 약간 감소
        model.add(LSTM(64, return_sequences=True)) 
        model.add(Dropout(0.4))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dropout(0.4))
        model.add(Dense(3, activation='softmax', dtype='float32'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_seq, y_seq, X_flat, y_flat):
        # 1. XGBoost 학습
        self.xgb_model = XGBClassifier(**config.XGB_PARAMS, num_class=3)
        
        # Class Imbalance 처리
        classes = np.unique(y_flat)
        if len(classes) > 0:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_flat)
            # weights가 비어있거나 매칭되지 않는 경우 방어 로직
            weight_dict = dict(zip(classes, weights))
            sample_weights = np.array([weight_dict.get(y, 1.0) for y in y_flat])
        else:
            sample_weights = None
            weight_dict = {}

        self.xgb_model.fit(X_flat, y_flat, sample_weight=sample_weights, verbose=False)
        
        # 2. LSTM 학습
        if self.lstm_model is None:
            self.lstm_model = self.build_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        # Early Stopping 추가
        es = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
        
        self.lstm_model.fit(
            X_seq, y_seq, 
            epochs=config.TRAIN_EPOCHS, 
            batch_size=config.BATCH_SIZE, 
            verbose=0, 
            class_weight=weight_dict if weight_dict else None,
            callbacks=[es]
        )

    def batch_predict(self, X_seq, X_flat):
        # 모델이 학습되지 않았을 경우 방어
        if not self.xgb_model or not self.lstm_model:
            return np.zeros(len(X_flat))

        # 확률 예측
        xgb_probs = self.xgb_model.predict_proba(X_flat)
        lstm_probs = self.lstm_model.predict(X_seq, batch_size=config.BATCH_SIZE, verbose=0)
        
        # 앙상블 (가중 평균)
        final_probs = (xgb_probs * config.W_XGB) + (lstm_probs * config.W_LSTM)
        
        # Score 계산 (Long확률 - Short확률)
        # [0]=Short, [1]=Neutral, [2]=Long
        return final_probs[:, 2] - final_probs[:, 0]