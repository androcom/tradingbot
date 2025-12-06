# ai_models.py
import numpy as np
import os
import joblib
import logging
import pandas as pd

# [성능 최적화 1] 불필요한 TF 로그 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import config

logger = logging.getLogger("BacktestLogger")

# [성능 최적화 2] GPU 메모리 동적 할당
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"   [System] GPU Memory Growth Enabled: {len(gpus)} GPUs")
    except RuntimeError as e:
        logger.error(f"   [System] GPU Setup Error: {e}")

# [성능 최적화 3] Mixed Precision
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    logger.info("   [System] Mixed Precision (FP16) Enabled")
except Exception as e:
    logger.warning(f"   [System] Failed to set Mixed Precision: {e}")

# [성능 최적화 4] XLA 활성화
tf.config.optimizer.set_jit(True)


class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total_epochs = self.params.get('epochs', config.TRAIN_EPOCHS)
        if epoch == 0 or epoch == (total_epochs - 1):
            logger.info(f"    Epoch {epoch+1}/{total_epochs} - loss: {logs.get('loss'):.4f} - accuracy: {logs.get('cls_output_accuracy'):.4f}")

class HybridEnsemble:
    def __init__(self, symbol):
        self.symbol = symbol
        safe_symbol = symbol.replace('/', '_')
        self.xgb_path = os.path.join(config.MODEL_DIR, f'xgb_{safe_symbol}.pkl')
        self.lstm_path = os.path.join(config.MODEL_DIR, f'lstm_{safe_symbol}.h5')

        # XGBoost 모델 정의
        self.xgb_model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            tree_method='hist', device='cuda',
            objective='multi:softprob', num_class=3, eval_metric='mlogloss'
        )
        self.lstm_model = None

    def save_models(self):
        joblib.dump(self.xgb_model, self.xgb_path)
        if self.lstm_model: self.lstm_model.save(self.lstm_path)

    def build_multi_output_lstm(self, input_shape):
        inputs = Input(shape=input_shape)
        
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        # Output 1: 방향 분류 (3 class)
        out_cls = Dense(3, activation='softmax', name='cls_output', dtype='float32')(x)
        
        # Output 2: 변동성 예측 (Regression)
        out_reg = Dense(1, activation='relu', name='reg_output', dtype='float32')(x)
        
        model = Model(inputs=inputs, outputs=[out_cls, out_reg])
        
        model.compile(
            optimizer='adam', 
            loss={'cls_output': 'sparse_categorical_crossentropy', 'reg_output': 'mse'},
            loss_weights={'cls_output': 1.0, 'reg_output': 0.5},
            metrics={'cls_output': 'accuracy'}
        )
        return model

    def train(self, X_seq, y_cls, y_vol, X_flat, y_flat_cls, features_name=None, is_update=False):
        # 1. XGBoost 학습
        classes = np.unique(y_flat_cls)
        if len(classes) > 1:
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_flat_cls)
            class_weight_dict = dict(zip(classes, weights))
            
            # Short(0) / Long(2)에 가중치 3배 부여
            if 0 in class_weight_dict: class_weight_dict[0] *= 3.0
            if 2 in class_weight_dict: class_weight_dict[2] *= 3.0
            
            sample_weights_xgb = np.array([class_weight_dict[y] for y in y_flat_cls])
            
            # [오류 수정된 부분] self.xgb_ -> self.xgb_model
            self.xgb_model.fit(X_flat, y_flat_cls, sample_weight=sample_weights_xgb)
        else:
             self.xgb_model.fit(X_flat, y_flat_cls)

        if features_name is not None and not is_update:
            try:
                importances = self.xgb_model.feature_importances_
                feature_imp = pd.DataFrame(sorted(zip(importances, features_name)), columns=['Value','Feature'])
                logger.info("\n[Model] Top 10 Important Features:")
                logger.info(feature_imp.sort_values(by="Value", ascending=False).head(10).to_string(index=False))
            except: pass

        # 2. LSTM 학습
        if self.lstm_model is None:
            self.lstm_model = self.build_multi_output_lstm((X_seq.shape[1], X_seq.shape[2]))
        
        # [수정] class_weight 오류 해결을 위해 sample_weight 사용
        # 분류(cls_output)에만 가중치 적용 (Short/Long 중요도 상향)
        cls_weight_map = {0: 3.0, 1: 1.0, 2: 3.0}
        sample_weights_cls = np.array([cls_weight_map.get(y, 1.0) for y in y_cls])
        
        logger.info(f"   >> LSTM Training ({config.TRAIN_EPOCHS} epochs)...")
        self.lstm_model.fit(
            X_seq, 
            {'cls_output': y_cls, 'reg_output': y_vol},
            epochs=config.TRAIN_EPOCHS, 
            batch_size=config.BATCH_SIZE, 
            verbose=0,
            # [핵심] 출력별 샘플 가중치 전달
            sample_weight={'cls_output': sample_weights_cls}, 
            callbacks=[LoggingCallback()]
        )
        self.save_models()

    def predict(self, X_seq, X_flat):
        xgb_probs = self.xgb_model.predict_proba(X_flat)
        xgb_score = xgb_probs[:, 2] - xgb_probs[:, 0]
        
        lstm_preds = self.lstm_model.predict(X_seq, batch_size=config.BATCH_SIZE, verbose=0)
        lstm_cls_probs = lstm_preds[0]
        lstm_vol_pred = lstm_preds[1].flatten()
        
        lstm_score = lstm_cls_probs[:, 2] - lstm_cls_probs[:, 0]
        
        final_score = (xgb_score * config.W_XGB) + (lstm_score * config.W_LSTM)
        
        return final_score, lstm_vol_pred