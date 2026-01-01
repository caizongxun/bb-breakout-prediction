import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM neural network for sequence prediction"""
    
    @staticmethod
    def create(seq_length, n_features, lstm_units=128):
        """
        Create LSTM model
        
        Args:
            seq_length: Number of time steps
            n_features: Number of features
            lstm_units: LSTM layer units
        """
        model = keras.Sequential([
            Input(shape=(seq_length, n_features)),
            LSTM(lstm_units, return_sequences=True, activation='relu'),
            Dropout(0.2),
            LSTM(lstm_units // 2, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC()])
        
        logger.info(f"Created LSTM model")
        return model


class TransformerModel:
    """Transformer model with attention mechanism"""
    
    @staticmethod
    def create(seq_length, n_features, num_heads=8, head_dim=64):
        """
        Create Transformer model
        """
        inputs = Input(shape=(seq_length, n_features))
        x = inputs
        
        # Transformer encoder blocks
        for _ in range(2):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=head_dim,
                dropout=0.1
            )(x, x)
            attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network
            ffn_output = Dense(256, activation='relu')(attention_output)
            ffn_output = Dense(n_features)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
        
        # Classification head
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC()])
        
        logger.info(f"Created Transformer model")
        return model


class EnsembleModel:
    """Ensemble of LSTM and XGBoost"""
    
    @staticmethod
    def create_lstm_feature_extractor(seq_length, n_features, encoding_dim=32):
        """
        LSTM for feature extraction
        """
        model = keras.Sequential([
            Input(shape=(seq_length, n_features)),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, activation='relu'),
            Dense(encoding_dim, activation='relu')
        ])
        
        logger.info(f"Created LSTM feature extractor")
        return model
    
    @staticmethod
    def create_xgboost_classifier():
        """
        XGBoost classifier
        """
        model = XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        logger.info(f"Created XGBoost classifier")
        return model


class AutoencoderFeatureLearner:
    """Autoencoder for unsupervised feature learning"""
    
    @staticmethod
    def create(seq_length, n_features, encoding_dim=16):
        """
        Create autoencoder for feature compression
        """
        # Encoder
        inputs = Input(shape=(seq_length, n_features))
        encoded = LSTM(encoding_dim, activation='relu')(inputs)
        
        # Decoder
        decoded = keras.layers.RepeatVector(seq_length)(encoded)
        decoded = keras.layers.LSTM(n_features, activation='linear', return_sequences=True)(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder for feature extraction
        encoder = keras.Model(inputs, encoded)
        
        logger.info(f"Created Autoencoder (input: {n_features}D, encoding: {encoding_dim}D)")
        return autoencoder, encoder
