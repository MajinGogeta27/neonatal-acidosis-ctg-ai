import numpy as np
import scipy.signal as signal
import scipy.stats as stats

class CTGFeatureExtractor:
    """
    Extracts clinical, statistical, spectral, and non-linear features from 4 Hz FHR & UC signals.
    Follows FIGO guidelines and clinical cardiotocography standards.
    """
    def __init__(self, sampling_rate=4.0):
        self.fs = sampling_rate

    def estimate_baseline_fhr(self, fhr):
        """
        Estimates baseline FHR using a 3-minute moving median filter.
        """
        win_size = int(3 * 60 * self.fs)
        if win_size % 2 == 0:
            win_size += 1
        win_size = min(win_size, len(fhr) if len(fhr)%2!=0 else len(fhr)-1)
        if win_size < 3:
            return np.median(fhr)
        baseline = signal.medfilt(fhr, kernel_size=win_size)
        return baseline

    def compute_stv_ltv(self, fhr):
        """
        Short-Term Variability (STV) & Long-Term Variability (LTV).
        STV: mean difference between successive 4 Hz samples over 1-minute blocks.
        LTV: difference between min and max FHR in 1-minute blocks.
        """
        block_samples = int(60 * self.fs) # 240 samples per minute
        num_blocks = len(fhr) // block_samples
        if num_blocks == 0:
            return np.mean(np.abs(np.diff(fhr))), np.std(fhr)

        stv_list = []
        ltv_list = []
        for i in range(num_blocks):
            block = fhr[i*block_samples : (i+1)*block_samples]
            # STV: mean sample-to-sample diff
            stv_block = np.mean(np.abs(np.diff(block)))
            # LTV: max - min in 1 min
            ltv_block = np.ptp(block)
            stv_list.append(stv_block)
            ltv_list.append(ltv_block)

        return np.mean(stv_list), np.mean(ltv_list)

    def detect_decelerations(self, fhr, baseline):
        """
        Detects decelerations (drops >= 15 bpm below baseline lasting >= 15 seconds).
        Returns deceleration metrics: count, max_depth, total_duration, total_auc, severe_count.
        """
        diff = baseline - fhr
        is_decel = diff >= 15.0 # threshold 15 bpm drop
        
        # Find contiguous regions
        padded = np.pad(is_decel.astype(int), (1, 1), mode='constant')
        starts = np.where((padded[1:] - padded[:-1]) == 1)[0]
        ends = np.where((padded[1:] - padded[:-1]) == -1)[0]

        min_samples = int(15.0 * self.fs) # 15 seconds = 60 samples
        decel_count = 0
        max_depth = 0.0
        total_duration = 0.0
        total_auc = 0.0
        severe_count = 0

        for s, e in zip(starts, ends):
            duration_samples = e - s
            if duration_samples >= min_samples:
                decel_count += 1
                duration_sec = duration_samples / self.fs
                total_duration += duration_sec
                region_diff = diff[s:e]
                depth = np.max(region_diff)
                if depth > max_depth:
                    max_depth = depth
                auc = np.sum(region_diff) / self.fs
                total_auc += auc

                # Severe deceleration: depth > 40 bpm or FHR < 100 bpm
                if depth > 40.0 or np.min(fhr[s:e]) < 100.0:
                    severe_count += 1

        return {
            'decel_count': decel_count,
            'max_decel_depth': max_depth,
            'total_decel_duration': total_duration,
            'total_decel_auc': total_auc,
            'severe_decel_count': severe_count
        }

    def detect_accelerations(self, fhr, baseline):
        """
        Detects accelerations (rises >= 15 bpm above baseline lasting >= 15 seconds).
        """
        diff = fhr - baseline
        is_accel = diff >= 15.0
        padded = np.pad(is_accel.astype(int), (1, 1), mode='constant')
        starts = np.where((padded[1:] - padded[:-1]) == 1)[0]
        ends = np.where((padded[1:] - padded[:-1]) == -1)[0]

        min_samples = int(15.0 * self.fs)
        accel_count = 0
        total_auc = 0.0

        for s, e in zip(starts, ends):
            if (e - s) >= min_samples:
                accel_count += 1
                total_auc += np.sum(diff[s:e]) / self.fs

        return {
            'accel_count': accel_count,
            'total_accel_auc': total_auc
        }

    def analyze_contractions(self, uc):
        """
        Analyzes Uterine Contractions (UC): peak detection, frequency per 10 min, mean duration, AUC.
        """
        # Smooth UC signal
        win_len = int(5 * self.fs) # 5-second smoothing
        if win_len % 2 == 0:
            win_len += 1
        uc_smooth = signal.medfilt(uc, kernel_size=min(win_len, len(uc) if len(uc)%2!=0 else len(uc)-1))
        
        # Prominent peaks (contractions)
        min_distance = int(60.0 * self.fs) # at least 1 min apart
        peaks, props = signal.find_peaks(uc_smooth, height=20.0, distance=min_distance)

        total_min = len(uc) / (self.fs * 60.0)
        contraction_freq_10m = (len(peaks) / total_min) * 10.0 if total_min > 0 else 0.0
        tachysystole = 1.0 if contraction_freq_10m > 5.0 else 0.0

        mean_amplitude = np.mean(props['peak_heights']) if len(peaks) > 0 else 0.0
        total_uc_auc = np.sum(uc_smooth) / self.fs

        return {
            'contraction_count': len(peaks),
            'contraction_freq_10m': contraction_freq_10m,
            'tachysystole': tachysystole,
            'mean_uc_amplitude': mean_amplitude,
            'total_uc_auc': total_uc_auc
        }

    def compute_spectral_nonlinear(self, fhr):
        """
        Spectral PSD power (LF: 0.04-0.15 Hz, HF: 0.15-0.4 Hz) and sample entropy.
        """
        freqs, psd = signal.welch(fhr, fs=self.fs, nperseg=min(1024, len(fhr)))
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        if trapz_fn is None:
            from scipy.integrate import trapezoid as trapz_fn

        lf_power = float(trapz_fn(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(trapz_fn(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else 0.0
        lf_hf_ratio = (lf_power / hf_power) if hf_power > 1e-6 else 0.0

        # Sample entropy approximation via signal standard deviation & diff ratios
        samp_ent = stats.differential_entropy(fhr) if hasattr(stats, 'differential_entropy') else np.std(fhr)

        return {
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'samp_entropy': samp_ent if not np.isnan(samp_ent) else 0.0
        }

    def extract_features(self, fhr, uc):
        """
        Extracts a dictionary of all domain-engineered features from 1D FHR & UC arrays.
        """
        baseline = self.estimate_baseline_fhr(fhr)
        mean_baseline = np.mean(baseline)

        # Statistical features
        mean_fhr = np.mean(fhr)
        std_fhr = np.std(fhr)
        min_fhr = np.min(fhr)
        max_fhr = np.max(fhr)
        range_fhr = max_fhr - min_fhr
        iqr_fhr = np.percentile(fhr, 75) - np.percentile(fhr, 25)
        skew_fhr = float(stats.skew(fhr))
        kurt_fhr = float(stats.kurtosis(fhr))

        # Variability
        stv, ltv = self.compute_stv_ltv(fhr)

        # Decelerations & Accelerations
        decel_feats = self.detect_decelerations(fhr, baseline)
        accel_feats = self.detect_accelerations(fhr, baseline)

        # Contractions
        uc_feats = self.analyze_contractions(uc)

        # Spectral & Non-linear
        spec_feats = self.compute_spectral_nonlinear(fhr)

        # Aggregate into feature dictionary
        features = {
            'mean_baseline_fhr': mean_baseline,
            'mean_fhr': mean_fhr,
            'std_fhr': std_fhr,
            'min_fhr': min_fhr,
            'max_fhr': max_fhr,
            'range_fhr': range_fhr,
            'iqr_fhr': iqr_fhr,
            'skew_fhr': skew_fhr if not np.isnan(skew_fhr) else 0.0,
            'kurt_fhr': kurt_fhr if not np.isnan(kurt_fhr) else 0.0,
            'stv': stv,
            'ltv': ltv,
            'samp_entropy': spec_feats['samp_entropy'],
            'lf_power': spec_feats['lf_power'],
            'hf_power': spec_feats['hf_power'],
            'lf_hf_ratio': spec_feats['lf_hf_ratio']
        }
        features.update(decel_feats)
        features.update(accel_feats)
        features.update(uc_feats)

        return features

    def extract_features_dataset(self, fhr_signals, uc_signals):
        """
        Extracts features for all samples in the dataset.
        Returns a 2D numpy array of shape (N, num_features) and feature names list.
        """
        print(f"Extracting features for {len(fhr_signals)} signals...")
        feature_list = []
        for i in range(len(fhr_signals)):
            feats = self.extract_features(fhr_signals[i], uc_signals[i])
            feature_list.append(feats)

        feature_names = list(feature_list[0].keys())
        X = np.zeros((len(feature_list), len(feature_names)), dtype=np.float32)
        for i, fdict in enumerate(feature_list):
            for j, name in enumerate(feature_names):
                val = fdict[name]
                X[i, j] = val if not np.isnan(val) and not np.isinf(val) else 0.0

        print(f"Extracted {X.shape[1]} features for {X.shape[0]} samples.")
        return X, feature_names

if __name__ == "__main__":
    extractor = CTGFeatureExtractor()
    fhr_dummy = 140.0 + 10.0 * np.sin(np.linspace(0, 50, 7200))
    uc_dummy = 10.0 + 30.0 * (np.sin(np.linspace(0, 20, 7200)) > 0.5)
    feats = extractor.extract_features(fhr_dummy, uc_dummy)
    print("Extracted dummy features count:", len(feats))
    for k, v in list(feats.items())[:10]:
        print(f"  {k}: {v:.3f}")
