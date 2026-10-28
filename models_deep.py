import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------
# Shared preprocessing (single source of truth for training AND saliency)
# ----------------------------------------------------------------------
FHR_MEAN, FHR_STD = 140.0, 20.0
UC_MEAN, UC_STD = 20.0, 25.0
PH_MIN, PH_MAX = 6.5, 7.6      # physiological range used for sanity checks
N_CHANNELS = 3                 # FHR, UC, valid-signal mask


def valid_length_mask(fhr, uc):
    """
    1 for samples up to the last point where FHR or UC changes, else 0.
    Detects flat forward-filled / padded tails. Works on [L] or [N, L].
    """
    fhr = np.asarray(fhr, dtype=np.float32)
    uc = np.asarray(uc, dtype=np.float32)
    changed = np.zeros(fhr.shape, dtype=bool)
    changed[..., 1:] = (fhr[..., 1:] != fhr[..., :-1]) | (uc[..., 1:] != uc[..., :-1])
    idx = np.arange(fhr.shape[-1])
    last = np.where(changed, idx, 0).max(axis=-1)
    return (idx <= np.expand_dims(last, -1)).astype(np.float32)


def prepare_input(fhr, uc):
    """
    Raw signals -> normalized, masked model input.
    [L] -> [3, L]   or   [N, L] -> [N, 3, L]
    """
    fhr = np.nan_to_num(np.asarray(fhr, dtype=np.float32), nan=FHR_MEAN)
    uc = np.nan_to_num(np.asarray(uc, dtype=np.float32), nan=UC_MEAN)
    mask = valid_length_mask(fhr, uc)
    fhr_n = (fhr - FHR_MEAN) / FHR_STD * mask
    uc_n = (uc - UC_MEAN) / UC_STD * mask
    return np.stack([fhr_n, uc_n, mask], axis=-2).astype(np.float32)


class CTGSignalDataset(Dataset):
    def __init__(self, fhr_signals, uc_signals, y_ph, y_710, y_705):
        self.x = torch.tensor(prepare_input(fhr_signals, uc_signals), dtype=torch.float32)
        self.y_ph = torch.tensor(y_ph, dtype=torch.float32)
        self.y_710 = torch.tensor(y_710, dtype=torch.float32)
        self.y_705 = torch.tensor(y_705, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_ph[idx], self.y_710[idx], self.y_705[idx]


class CTGBase(nn.Module):
    """Holds pH de-standardization so model(x) always returns real pH."""
    def __init__(self):
        super().__init__()
        self.register_buffer("ph_mean", torch.tensor(7.2))
        self.register_buffer("ph_std", torch.tensor(0.1))

    def set_ph_scale(self, mean, std):
        self.ph_mean.fill_(float(mean))
        self.ph_std.fill_(float(max(std, 1e-3)))

    def to_ph(self, raw):
        return self.ph_mean + self.ph_std * raw


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, pool=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CTG1DCNN(CTGBase):
    def __init__(self, in_channels=N_CHANNELS, hidden_dim=64):
        super().__init__()
        self.b1 = Conv1DBlock(in_channels, 32, kernel_size=15)
        self.b2 = Conv1DBlock(32, 64, kernel_size=11)
        self.b3 = Conv1DBlock(64, 128, kernel_size=7)
        self.b4 = Conv1DBlock(128, 128, kernel_size=5)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_shared = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.head_ph = nn.Linear(hidden_dim, 1)
        self.head_710 = nn.Linear(hidden_dim, 1)
        self.head_705 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        feat = self.b4(self.b3(self.b2(self.b1(x))))
        feat = self.global_pool(feat).squeeze(-1)
        shared = self.fc_shared(feat)
        ph_pred = self.to_ph(self.head_ph(shared).squeeze(-1))
        return ph_pred, self.head_710(shared).squeeze(-1), self.head_705(shared).squeeze(-1), feat


class CTGCNNLSTM(CTGBase):
    def __init__(self, in_channels=N_CHANNELS, hidden_dim=64):
        super().__init__()
        self.conv_stem = nn.Sequential(
            Conv1DBlock(in_channels, 32, kernel_size=15, pool=True),
            Conv1DBlock(32, 64, kernel_size=7, pool=True),
            Conv1DBlock(64, 64, kernel_size=5, pool=True),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc_shared = nn.Sequential(nn.Linear(128, hidden_dim), nn.ReLU(), nn.Dropout(0.3))
        self.head_ph = nn.Linear(hidden_dim, 1)
        self.head_710 = nn.Linear(hidden_dim, 1)
        self.head_705 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        conv_feats = self.conv_stem(x).transpose(1, 2)
        lstm_out, _ = self.lstm(conv_feats)
        feat = torch.mean(lstm_out, dim=1)
        shared = self.fc_shared(feat)
        ph_pred = self.to_ph(self.head_ph(shared).squeeze(-1))
        return ph_pred, self.head_710(shared).squeeze(-1), self.head_705(shared).squeeze(-1), feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CTGTransformer(CTGBase):
    def __init__(self, in_channels=N_CHANNELS, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.conv_subsample = nn.Sequential(
            Conv1DBlock(in_channels, 32, kernel_size=15, pool=True),
            Conv1DBlock(32, d_model, kernel_size=7, pool=True),
        )
        self.pos_encoder = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128,
                                           dropout=0.2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_shared = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.3))
        self.head_ph = nn.Linear(d_model, 1)
        self.head_710 = nn.Linear(d_model, 1)
        self.head_705 = nn.Linear(d_model, 1)

    def forward(self, x):
        sub = self.conv_subsample(x).transpose(1, 2)
        sub = self.pos_encoder(sub)
        feat = torch.mean(self.transformer_encoder(sub), dim=1)
        shared = self.fc_shared(feat)
        ph_pred = self.to_ph(self.head_ph(shared).squeeze(-1))
        return ph_pred, self.head_710(shared).squeeze(-1), self.head_705(shared).squeeze(-1), feat


class DeepModelTrainer:
    def __init__(self, n_splits=5, epochs=25, lr=1e-3, device=None, verbose_every=1):
        self.n_splits = n_splits
        self.epochs = epochs
        self.lr = lr
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose_every = verbose_every
        self.fold_val_indices = []   # val indices per fold, aligned with returned models

    @staticmethod
    def regression_stats(y, p):
        err = np.asarray(p) - np.asarray(y)
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
        }

    def _predict(self, model, loader):
        model.eval()
        ph, p710, p705 = [], [], []
        with torch.no_grad():
            for bx, _, _, _ in loader:
                a, b, c, _ = model(bx.to(self.device))
                ph.append(a.cpu().numpy())
                p710.append(torch.sigmoid(b).cpu().numpy())
                p705.append(torch.sigmoid(c).cpu().numpy())
        return np.concatenate(ph), np.concatenate(p710), np.concatenate(p705)

    @staticmethod
    def _pos_weight(y, device, cap=10.0):
        pos = float(np.sum(y))
        neg = float(len(y) - pos)
        return torch.tensor(min(neg / max(pos, 1.0), cap), dtype=torch.float32, device=device)

    def train_evaluate(self, model_class, fhr_signals, uc_signals, y_ph, y_710, y_705, model_name="1D-CNN"):
        print(f"\n--- Running {self.n_splits}-Fold CV for PyTorch {model_name} on Device: {self.device} ---")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        oof_ph = np.zeros(len(y_ph), dtype=np.float32)
        oof_710 = np.zeros(len(y_710), dtype=np.float32)
        oof_705 = np.zeros(len(y_705), dtype=np.float32)
        models_list = []
        self.fold_val_indices = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(fhr_signals, y_710)):
            torch.manual_seed(42 + fold)

            train_ds = CTGSignalDataset(fhr_signals[tr_idx], uc_signals[tr_idx],
                                        y_ph[tr_idx], y_710[tr_idx], y_705[tr_idx])
            val_ds = CTGSignalDataset(fhr_signals[va_idx], uc_signals[va_idx],
                                      y_ph[va_idx], y_710[va_idx], y_705[va_idx])
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

            model = model_class().to(self.device)
            model.set_ph_scale(np.mean(y_ph[tr_idx]), np.std(y_ph[tr_idx]))
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            crit_710 = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(y_710[tr_idx], self.device))
            crit_705 = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(y_705[tr_idx], self.device))

            base_mae = np.mean(np.abs(y_ph[va_idx] - np.mean(y_ph[tr_idx])))
            print(f"\n  Fold {fold + 1}/{self.n_splits} | train n={len(tr_idx)}, val n={len(va_idx)} "
                  f"| baseline (predict train mean) val MAE = {base_mae:.3f}")

            for epoch in range(1, self.epochs + 1):
                model.train()
                run_loss, n_seen = 0.0, 0
                tr_pred, tr_true = [], []
                for bx, by_ph, by_710, by_705 in train_loader:
                    bx, by_ph = bx.to(self.device), by_ph.to(self.device)
                    by_710, by_705 = by_710.to(self.device), by_705.to(self.device)

                    optimizer.zero_grad()
                    pred_ph, l710, l705, _ = model(bx)

                    # Huber on standardized pH
                    loss_ph = F.smooth_l1_loss(pred_ph / model.ph_std, by_ph / model.ph_std)
                    loss = loss_ph + 0.5 * crit_710(l710, by_710) + 0.5 * crit_705(l705, by_705)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    run_loss += loss.item() * len(bx)
                    n_seen += len(bx)
                    tr_pred.append(pred_ph.detach().cpu().numpy())
                    tr_true.append(by_ph.cpu().numpy())
                scheduler.step()

                if epoch % self.verbose_every == 0 or epoch == self.epochs:
                    tr = self.regression_stats(np.concatenate(tr_true), np.concatenate(tr_pred))
                    v_ph, _, _ = self._predict(model, val_loader)
                    va = self.regression_stats(y_ph[va_idx], v_ph)
                    n_oor = int(np.sum((v_ph < PH_MIN) | (v_ph > PH_MAX)))
                    print(f"    Ep {epoch:02d}/{self.epochs} | loss {run_loss / n_seen:.4f} "
                          f"| train MAE {tr['mae']:.3f} "
                          f"| val MAE {va['mae']:.3f} RMSE {va['rmse']:.3f} bias {va['bias']:+.3f} "
                          f"| out-of-range {n_oor}")

            v_ph, v_710, v_705 = self._predict(model, val_loader)
            oof_ph[va_idx], oof_710[va_idx], oof_705[va_idx] = v_ph, v_710, v_705
            models_list.append(model)
            self.fold_val_indices.append(np.asarray(va_idx))

        n_oor = int(np.sum((oof_ph < PH_MIN) | (oof_ph > PH_MAX)))
        oof_ph = np.clip(oof_ph, PH_MIN, PH_MAX)
        s = self.regression_stats(y_ph, oof_ph)
        base = np.mean(np.abs(y_ph - np.mean(y_ph)))
        print(f"\n  [{model_name}] OOF pH: MAE {s['mae']:.3f} | RMSE {s['rmse']:.3f} | bias {s['bias']:+.3f} "
              f"| baseline MAE {base:.3f} | raw predictions outside [{PH_MIN}, {PH_MAX}]: {n_oor} (clipped)")

        return oof_ph, oof_710, oof_705, models_list


if __name__ == "__main__":
    fhr_dummy = np.random.randn(20, 7200) + 140.0
    uc_dummy = np.random.randn(20, 7200) + 20.0
    y_ph_dummy = 7.2 + 0.1 * np.random.randn(20)
    y_710_dummy = (y_ph_dummy < 7.15).astype(int)
    y_705_dummy = (y_ph_dummy < 7.05).astype(int)
    trainer = DeepModelTrainer(n_splits=2, epochs=2)
    oof_ph, _, _, _ = trainer.train_evaluate(CTG1DCNN, fhr_dummy, uc_dummy, y_ph_dummy,
                                             y_710_dummy, y_705_dummy, "1D-CNN Test")
    print("OOF pH shape:", oof_ph.shape)