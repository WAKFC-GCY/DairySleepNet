"""Build the research STFT representation and save explicit subject boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np

from args import Config, ROOT, canonical_modality


def stft_29x128(x_epoch, fs=100, win_sec=2.0, overlap_sec=1.0, nfft=256, pad_size=29):
    """Hann window, FFT bins 0..127, log1p magnitude; matches the research code."""
    x_epoch = np.asarray(x_epoch, dtype=np.float32)
    if x_epoch.ndim != 2 or not np.isfinite(x_epoch).all():
        raise ValueError("Expected finite signals of shape (epochs, samples)")
    win = int(win_sec * fs)
    hop = int((win_sec - overlap_sec) * fs)
    if hop <= 0 or nfft < win or x_epoch.shape[1] < win:
        raise ValueError("Invalid STFT window, hop, FFT size or epoch length")
    window = np.hanning(win).astype(np.float32)
    frames = list(range(0, x_epoch.shape[1] - win + 1, hop))
    out = np.zeros((len(x_epoch), pad_size, nfft // 2), dtype=np.float32)
    for n, x in enumerate(x_epoch):
        for t, start in enumerate(frames[:pad_size]):
            fft = np.fft.rfft(x[start:start + win] * window, n=nfft)
            out[n, t] = np.abs(fft[:nfft // 2]).astype(np.float32)
    return np.log1p(out)


def prepare_tf(data_root, config):
    data_root = Path(data_root)
    raw = data_root / "raw" / config.modality
    output = data_root / "tf" / config.modality
    if (output / "manifest.json").exists():
        raise FileExistsError(f"{output} already has a manifest; use a fresh --data-root")
    subjects = sorted(p.stem for p in (raw / "labels").glob("*.npy"))
    if not subjects:
        raise FileNotFoundError(f"No labels in {raw / 'labels'}")
    output.mkdir(parents=True, exist_ok=True)
    records, offset = [], 0
    for subject in subjects:
        labels = np.load(raw / "labels" / f"{subject}.npy", allow_pickle=False)
        if labels.ndim != 1 or len(labels) == 0 or not np.isin(labels, [0, 1, 2]).all():
            raise ValueError(f"{subject}: invalid prepared labels")
        records.append({"id": subject, "start": offset, "end": offset + len(labels)})
        offset += len(labels)
    stats = {}
    for channel in config.channel_list:
        parts = []
        for record in records:
            subject = record["id"]
            x = np.load(raw / "signals" / f"{subject}__{channel}.npy", allow_pickle=False)
            if x.shape != (record["end"] - record["start"],
                           int(config.target_sfreq * config.epoch_duration)):
                raise ValueError(f"{subject}/{channel}: signal and label lengths differ")
            parts.append(stft_29x128(x, config.target_sfreq, config.win_size_sec,
                                    config.overlap_sec, config.nfft, config.pad_size))
        values = np.concatenate(parts, axis=0)
        # Preserve all-epoch normalization from the published experiment pipeline.
        mean, std = values.mean(), values.std() + 1e-6
        values = ((values - mean) / std).astype(np.float32)
        np.save(output / f"TF_{channel}_mean_std.npy", values, allow_pickle=False)
        stats[channel] = {"mean": float(mean), "std": float(std)}
        print(f"{channel}: {values.shape}")
    manifest = {"modality": config.modality, "channels": config.channel_list,
                "feature_shape": [config.pad_size, config.dim_model], "subjects": records,
                "normalization": "global_per_channel_all_prepared_epochs", "statistics": stats}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--modality", type=canonical_modality, default="PSG")
    args = parser.parse_args()
    prepare_tf(args.data_root, Config(args.modality))


if __name__ == "__main__":
    main()
