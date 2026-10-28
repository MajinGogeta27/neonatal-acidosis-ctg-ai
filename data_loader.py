import os
import glob
import re
import numpy as np
import pandas as pd

class CTUDataLoader:
    """
    DataLoader for CTU-CHB Intrapartum Cardiotocography Database.
    Loads 4 Hz FHR and UC waveforms along with matched umbilical cord blood gas outcomes.
    """
    def __init__(self, data_dir=r"c:\Users\Timic\CTG Fetus\ctu-chb-intrapartum-cardiotocography-database-1.0.0", sampling_rate=4.0):
        self.data_dir = data_dir
        self.fs = sampling_rate

    def load_header_metadata(self, fpath):
        rec_id = os.path.basename(fpath).replace(".hea", "")
        info = {"rec_id": rec_id}
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # First line contains signal details
        header_first = lines[0].split()
        num_signals = int(header_first[1]) if len(header_first) > 1 else 2
        fs = float(header_first[2]) if len(header_first) > 2 else 4.0
        num_samples = int(header_first[3]) if len(header_first) > 3 else 0
        info['num_signals'] = num_signals
        info['fs'] = fs
        info['num_samples'] = num_samples

        for line in lines:
            if line.startswith("#"):
                match = re.match(r"^#\s*([A-Za-z0-9\.\(\)\/\s_\-]+?)\s{2,}(.+)$", line)
                if not match:
                    match = re.match(r"^#\s*(pH|BDecf|pCO2|BE|Apgar1|Apgar5|NICU days|Seizures|HIE|Intubation|Main diag\.|Other diag\.|Gest\. weeks|Weight\(g\)|Sex|Age|Gravidity|Parity|Diabetes|Hypertension|Preeclampsia|Liq\. praecox|Pyrexia|Meconium|Presentation|Induced|I\.stage|NoProgress|CK/KP|II\.stage|Deliv\. type|dbID|Rec\. type|Pos\. II\.st\.|Sig2Birth)\s+(.+)$", line)
                if match:
                    key = match.group(1).strip()
                    val = match.group(2).strip()
                    info[key] = val
        
        # Standardize numeric types
        def to_float(k):
            try:
                return float(info[k]) if (k in info and info[k] not in ['None', '']) else np.nan
            except Exception:
                return np.nan

        info['pH'] = to_float('pH')
        info['BDecf'] = to_float('BDecf')
        info['BE'] = to_float('BE')
        info['pCO2'] = to_float('pCO2')
        info['Apgar1'] = to_float('Apgar1')
        info['Apgar5'] = to_float('Apgar5')
        info['Gest_weeks'] = to_float('Gest. weeks')
        info['Weight_g'] = to_float('Weight(g)')
        info['Mother_age'] = to_float('Age')
        info['Stage_II_duration'] = to_float('II.stage')
        info['Deliv_type'] = to_float('Deliv. type')
        info['Induced'] = to_float('Induced')

        if not np.isnan(info['pH']):
            info['pH_less_710'] = 1 if info['pH'] < 7.10 else 0
            info['pH_less_705'] = 1 if info['pH'] < 7.05 else 0
        else:
            info['pH_less_710'] = np.nan
            info['pH_less_705'] = np.nan

        return info

    def load_signal(self, rec_id):
        """
        Loads signal from .dat file (Format 16 little-endian, 2 channels multiplexed: FHR, UC).
        """
        dat_path = os.path.join(self.data_dir, f"{rec_id}.dat")
        if not os.path.exists(dat_path):
            raise FileNotFoundError(f"Dat file not found: {dat_path}")

        # Try using wfdb if available
        try:
            import wfdb
            record = wfdb.rdrecord(os.path.join(self.data_dir, rec_id))
            signal = record.p_signal
            fhr = signal[:, 0]
            uc = signal[:, 1]
            return fhr, uc
        except Exception:
            # Binary fallback reading Format 16 int16
            raw_data = np.fromfile(dat_path, dtype=np.int16)
            raw_data = raw_data.reshape(-1, 2)
            fhr = raw_data[:, 0].astype(np.float32)
            uc = raw_data[:, 1].astype(np.float32)
            return fhr, uc

    def preprocess_signal(self, fhr, uc, max_gap_seconds=15.0):
        """
        Cleans zero dropouts in FHR using interpolation up to max_gap_seconds.
        Returns cleaned FHR, original UC, and signal quality (valid percentage).
        """
        fhr = fhr.copy()
        uc = uc.copy()
        
        # Valid FHR is typically between 50 and 220 bpm. 0 indicates dropout.
        invalid_mask = (fhr < 50.0) | (fhr > 220.0)
        valid_ratio = 1.0 - (np.sum(invalid_mask) / float(len(fhr)))

        # Spline/linear interpolation over short gaps
        fhr_clean = fhr.copy()
        fhr_clean[invalid_mask] = np.nan

        # Linear interpolate NaNs
        nans = np.isnan(fhr_clean)
        if np.all(nans):
            fhr_clean[:] = 120.0 # fallback default baseline
        elif np.any(nans):
            x = lambda z: z.nonzero()[0]
            fhr_clean[nans] = np.interp(x(nans), x(~nans), fhr_clean[~nans])

        return fhr_clean, uc, valid_ratio

    def get_window(self, fhr, uc, window_minutes=30.0):
        """
        Extracts the specified window_minutes prior to delivery (end of signal).
        """
        n_samples = int(window_minutes * 60.0 * self.fs)
        if len(fhr) >= n_samples:
            return fhr[-n_samples:], uc[-n_samples:]
        else:
            # Zero-pad at beginning if shorter
            pad_len = n_samples - len(fhr)
            fhr_padded = np.pad(fhr, (pad_len, 0), mode='edge')
            uc_padded = np.pad(uc, (pad_len, 0), mode='constant', constant_values=0)
            return fhr_padded, uc_padded

    def load_dataset(self, window_minutes=30.0):
        """
        Loads all dataset records, preprocessed signals, metadata, and ground truth outcomes.
        Returns a DataFrame of metadata and lists of signal windows.
        """
        hea_files = sorted(glob.glob(os.path.join(self.data_dir, "*.hea")))
        metadata_list = []
        fhr_signals = []
        uc_signals = []

        print(f"Loading CTU-CHB dataset ({len(hea_files)} records)...")
        for fpath in hea_files:
            meta = self.load_header_metadata(fpath)
            rec_id = meta['rec_id']
            try:
                fhr, uc = self.load_signal(rec_id)
                fhr_clean, uc_clean, valid_ratio = self.preprocess_signal(fhr, uc)
                fhr_win, uc_win = self.get_window(fhr_clean, uc_clean, window_minutes=window_minutes)

                meta['valid_ratio'] = valid_ratio
                meta['signal_len_min'] = len(fhr) / (self.fs * 60.0)

                metadata_list.append(meta)
                fhr_signals.append(fhr_win)
                uc_signals.append(uc_win)
            except Exception as e:
                print(f"Warning: Failed to load signal for record {rec_id}: {e}")

        df_meta = pd.DataFrame(metadata_list)
        fhr_signals = np.array(fhr_signals, dtype=np.float32)
        uc_signals = np.array(uc_signals, dtype=np.float32)

        print(f"Successfully loaded {len(df_meta)} records.")
        print(f"Signal matrix shape: FHR {fhr_signals.shape}, UC {uc_signals.shape}")
        return df_meta, fhr_signals, uc_signals

if __name__ == "__main__":
    loader = CTUDataLoader()
    df_meta, fhr, uc = loader.load_dataset(window_minutes=30.0)
    print(df_meta[['rec_id', 'pH', 'pH_less_710', 'pH_less_705', 'valid_ratio']].head())
