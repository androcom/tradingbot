# ai_models.py
import numpy as np
import os
import joblib
import logging
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback
from sklearn.utils.class_weight import compute_class_weight
import config

logger = logging.getLogger("BacktestLogger")

# 학습 과정 중 로그를 출력하기 위한 콜백 클래스
class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total_epochs = self.params.get('epochs', config.TRAIN_EPOCHS)
        
        # 학습의 시작(첫 에포크)과 끝(마지막 에포크)에만 로그를 출력하여 로그 양 조절
        if epoch == 0 or epoch == (total_epochs - 1):
            logger.info(f"    Epoch {epoch+1}/{total_epochs} - loss: {logs.get('loss'):.4f} - accuracy: {logs.get('accuracy'):.4f}")

# XGBoost와 LSTM을 결합한 하이브리드 앙상블 모델 클래스
class HybridEnsemble:
    def __init__(self, symbol):
        self.symbol = symbol
        safe_symbol = symbol.replace('/', '_')
        
        # 모델 파일 저장 경로 설정
        self.xgb_path = os.path.join(config.MODEL_DIR, f'xgb_{safe_symbol}.pkl')
        self.lstm_path = os.path.join(config.MODEL_DIR, f'lstm_{safe_symbol}.h5')

        # XGBoost 모델 초기화 (이전 학습된 모델을 불러오지 않고 새로 생성)
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05,
            tree_method='hist', device='cuda',
            objective='multi:softprob', num_class=3, eval_metric='mlogloss'
        )
        self.lstm_model = None

    # 학습된 모델들을 파일로 저장
    def save_models(self):
        joblib.dump(self.xgb_model, self.xgb_path)
        if self.lstm_model: self.lstm_model.save(self.lstm_path)

    # LSTM 모델 구조 정의 및 컴파일
    def build_lstm(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # 모델 학습 (XGBoost 및 LSTM)
    def train(self, X_seq, y_seq, X_flat, y_flat, is_update=False):
        # 클래스 불균형 처리를 위한 가중치 계산
        classes = np.unique(y_flat)
        if len(classes) < 2:
            class_weight_dict = {c: 1.0 for c in classes}
            sample_weights = np.ones(len(y_flat))
        else:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_flat)
            class_weight_dict = dict(zip(classes, weights))
            sample_weights = np.array([class_weight_dict[y] for y in y_flat])

        # XGBoost 모델 학습
        self.xgb_model.fit(X_flat, y_flat, sample_weight=sample_weights)
        
        # LSTM 모델이 없으면 생성
        if self.lstm_model is None:
            self.lstm_model = self.build_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        epochs = config.TRAIN_EPOCHS
        
        logger.info(f"   >> LSTM Training ({epochs} epochs)...")
        # LSTM 모델 학습
        self.lstm_model.fit(
            X_seq, y_seq, epochs=epochs, batch_size=config.BATCH_SIZE, 
            verbose=0, callbacks=[LoggingCallback()], class_weight=class_weight_dict
        )
        # 학습 완료 후 모델 저장
        self.save_models()

    # 배치 예측 수행 (XGBoost와 LSTM 결과 앙상블)
    def batch_predict(self, X_seq, X_flat, tech_scores):
        # XGBoost 예측 확률 계산
        xgb_probs = self.xgb_model.predict_proba(X_flat)
        # 상승 확률(인덱스 2)에서 하락 확률(인덱스 0)을 뺀 점수 계산
        xgb_score = xgb_probs[:, 2] - xgb_probs[:, 0]
        
        # LSTM 예측 확률 계산
        lstm_probs = self.lstm_model.predict(X_seq, batch_size=config.BATCH_SIZE, verbose=0)
        # 상승 확률(인덱스 2)에서 하락 확률(인덱스 0)을 뺀 점수 계산
        lstm_score = lstm_probs[:, 2] - lstm_probs[:, 0]
        
        # 두 모델의 점수를 가중 합산하여 최종 점수 도출
        final_scores = (
            (xgb_score * config.W_XGB) +
            (lstm_score * config.W_LSTM)
        )
        return final_scores
