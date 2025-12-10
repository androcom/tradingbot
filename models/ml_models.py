import numpy as np
import joblib
import os
import logging
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.utils.class_weight import compute_class_weight
import config

class FirstLastLogger(Callback):
    def on_train_begin(self, logs=None):
        logging.info(f"   [LSTM] Training Start (Max Epochs: {self.params['epochs']})")
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 10 == 0:
            logging.info(f"   [LSTM] Epoch {epoch+1:2d}: loss={logs.get('loss'):.4f}, acc={logs.get('accuracy'):.4f}")
    def on_train_end(self, logs=None):
        logging.info("   [LSTM] Training Finished.")

class HybridLearner:
    def __init__(self, model_dir):
        self.xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
        self.lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        self.xgb_model = None
        self.lstm_model = None

    def _build_lstm(self, input_shape):
        # [Config] 파라미터 로드
        params = config.LSTM_PARAMS
        model = Sequential([
            Input(shape=input_shape),
            LSTM(params['units_1'], return_sequences=True),
            Dropout(params['dropout']),
            LSTM(params['units_2']),
            Dropout(params['dropout']),
            Dense(3, activation='softmax', dtype='float32')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_flat, y_flat, X_seq, y_seq):
        # 1. XGBoost Train
        logging.info(">> Training XGBoost...")
        classes = np.unique(y_flat)
        weights = compute_class_weight('balanced', classes=classes, y=y_flat)
        sample_weights = np.array([dict(zip(classes, weights))[y] for y in y_flat])
        
        # [Config] XGB 파라미터 언패킹
        self.xgb_model = XGBClassifier(**config.XGB_PARAMS)
        self.xgb_model.fit(X_flat, y_flat, sample_weight=sample_weights)
        joblib.dump(self.xgb_model, self.xgb_path)

        # 2. LSTM Train
        logging.info(">> Training LSTM...")
        self.lstm_model = self._build_lstm((X_seq.shape[1], X_seq.shape[2]))
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        
        self.lstm_model.fit(
            X_seq, y_seq, 
            epochs=config.ML_EPOCHS, 
            batch_size=config.ML_BATCH_SIZE, 
            callbacks=[es, FirstLastLogger()],
            verbose=0
        )
        self.lstm_model.save(self.lstm_path)

    def load(self):
        if os.path.exists(self.xgb_path) and os.path.exists(self.lstm_path):
            self.xgb_model = joblib.load(self.xgb_path)
            self.lstm_model = load_model(self.lstm_path)
            return True
        return False

    def predict_proba(self, X_flat, X_seq):
        if not self.xgb_model or not self.lstm_model:
            if not self.load(): raise Exception("Models not loaded")
            
        xgb_p = self.xgb_model.predict_proba(X_flat)
        lstm_p = self.lstm_model.predict(X_seq, verbose=0)
        
        final_prob = (xgb_p + lstm_p) / 2
        signal = final_prob[:, 2] - final_prob[:, 0]
        return signal