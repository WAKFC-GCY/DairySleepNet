"""Model and preprocessing settings used by DairySleepNet."""

from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PSG_CHANNELS = ("Abdomen", "F3", "C3", "F4", "C4", "E1", "E2", "ChinL", "ChinR")
ACC_CHANNELS = ("Pressure", "Move_X", "Move_Y", "Move_Z")
CLASS_NAMES = ("Wake", "Sleep", "Rumination")


def canonical_modality(value):
    value = value.upper().replace("+", "_")
    value = {"PGS": "PSG", "PGS_ACC": "PSG_ACC", "RWS": "ACC",
             "PSG_RWS": "PSG_ACC"}.get(value, value)
    if value not in ("PSG", "ACC", "PSG_ACC"):
        raise ValueError("Modality must be PSG, ACC or PSG_ACC")
    return value


@dataclass
class Config:
    modality: str = "PSG"
    num_classes: int = 3
    batch_size: int = 32
    epochs: int = 200
    lr: float = 5e-6
    weight_decay: float = 1e-2
    patience: int = 20
    random_state: int = 42
    validation_fraction: float = 0.1
    label_smoothing: float = 0.1
    target_sfreq: int = 100
    epoch_duration: float = 30.0
    win_size_sec: float = 2.0
    overlap_sec: float = 1.0
    nfft: int = 256
    pad_size: int = 29
    dim_model: int = 128
    dropout: float = 0.2
    num_encoder: int = 4
    num_encoder_multi: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2

    def __post_init__(self):
        self.modality = canonical_modality(self.modality)

    @property
    def channel_list(self):
        if self.modality == "PSG":
            return list(PSG_CHANNELS)
        if self.modality == "ACC":
            return list(ACC_CHANNELS)
        return list(PSG_CHANNELS + ACC_CHANNELS)

    @property
    def num_channels(self):
        return len(self.channel_list)

    def to_dict(self):
        return asdict(self)
