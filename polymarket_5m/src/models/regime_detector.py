import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Deteksi regime pasar (volatilitas rendah/normal/tinggi).
    KRITIS: scaler dan classifier hanya di-fit pada training data SEKALI.
    """
    def __init__(self, n_regimes: int = 3):
        self.n_regimes    = n_regimes
        self.scaler       = None  # Di-fit SEKALI pada train
        self.classifier   = None  # Di-fit SEKALI pada train
        self.is_fitted    = False

    def _extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features untuk deteksi regime. HANYA dari data historis."""
        features = {}
        # Gunakan fitur yang tersedia di final_features.parquet
        if 'win_rvol' in df.columns:
            features['rvol'] = df['win_rvol']
        elif 'win_rvol_300' in df.columns:
            features['rvol'] = df['win_rvol_300']
            
        if 'snap_atr_14_norm' in df.columns:
            features['atr'] = df['snap_atr_14_norm']
        if 'snap_spread_pct' in df.columns:
            features['spread'] = df['snap_spread_pct']
        if 'snap_bb_width_20' in df.columns:
            features['bb_width'] = df['snap_bb_width_20']
        
        if 'past_rvol_8' in df.columns:
            features['past_rvol_8'] = df['past_rvol_8']
        elif 'past_rvol_4' in df.columns:
            features['past_rvol_4'] = df['past_rvol_4']

        return pd.DataFrame(features, index=df.index).fillna(0)

    def fit(self, X_train: pd.DataFrame) -> 'RegimeDetector':
        """
        HANYA dipanggil SEKALI dengan training data.
        """
        regime_features = self._extract_regime_features(X_train)
        if regime_features.empty:
            logger.warning("No regime features found. KMeans will not be effective.")
            return self

        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(regime_features)

        self.classifier = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        self.classifier.fit(features_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediksi regime untuk data baru.
        HANYA menggunakan scaler.transform() (TIDAK fit_transform()).
        """
        if not self.is_fitted:
            logger.warning("RegimeDetector belum di-fit. Mengembalikan regime 0 untuk semua.")
            return np.zeros(len(X))

        regime_features = self._extract_regime_features(X)
        if regime_features.empty:
            return np.zeros(len(X))
            
        # TRANSFORM SAJA — tidak fit ulang
        features_scaled = self.scaler.transform(regime_features)
        return self.classifier.predict(features_scaled)

    def save(self, path: str):
        joblib.dump({'scaler': self.scaler, 'classifier': self.classifier,
                     'n_regimes': self.n_regimes, 'is_fitted': self.is_fitted}, path)

    @classmethod
    def load(cls, path: str) -> 'RegimeDetector':
        data = joblib.load(path)
        obj  = cls(n_regimes=data['n_regimes'])
        obj.scaler     = data['scaler']
        obj.classifier = data['classifier']
        obj.is_fitted  = data['is_fitted']
        return obj
