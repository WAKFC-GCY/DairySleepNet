"""Convert aligned EDF recordings and numeric labels to 30-second epochs."""

import argparse
import json
from pathlib import Path

import mne
import numpy as np

from args import Config, PSG_CHANNELS, ACC_CHANNELS, ROOT, canonical_modality


def map_5to3(labels):
    """0=Wake; 1/2/3=Sleep; 4=Rumination; -1=missing (study coding)."""
    labels = np.atleast_1d(labels)
    if labels.ndim != 1 or not np.isin(labels, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("Labels must be a vector containing only -1, 0, 1, 2, 3, 4")
    labels = labels.astype(np.int64, copy=True)
    labels[np.isin(labels, [1, 2, 3])] = 1
    labels[labels == 4] = 2
    return labels


def read_channels(edf_path, channels, config, allow_missing):
    if not edf_path.is_file():
        raise FileNotFoundError(edf_path)
    with mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR") as raw:
        present = [ch for ch in channels if ch in raw.ch_names]
        missing = [ch for ch in channels if ch not in present]
        if missing and not allow_missing:
            raise ValueError(f"{edf_path.name}: missing channels {missing}; "
                             "use --allow-missing-channels to explicitly zero-fill")
        if not present:
            raise ValueError(f"{edf_path.name}: none of the requested channels is present")
        raw.pick(present)
        if not np.isclose(raw.info["sfreq"], config.target_sfreq):
            raw.resample(config.target_sfreq)
        values = raw.get_data().astype(np.float32)
    length = int(config.target_sfreq * config.epoch_duration)
    epochs = values.shape[1] // length
    values = values[:, :epochs * length].reshape(len(present), epochs, length)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite signals in {edf_path}")
    if missing:
        print(f"{edf_path.name}: zero-filling {missing}")
    return dict(zip(present, values)), epochs, missing


def prepare_dataset(input_dir, data_root, config, label_format="five", allow_missing=False):
    input_dir, data_root = Path(input_dir), Path(data_root)
    label_files = sorted(input_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label .txt files in {input_dir}")
    output = data_root / "raw" / config.modality
    # Reject mixed old/new preprocessing rather than silently retaining stale subjects.
    if (output / "labels").exists() and any((output / "labels").glob("*.npy")):
        raise FileExistsError(f"{output} already contains labels; use a fresh --data-root")
    (output / "labels").mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir(exist_ok=True)
    metadata = {"modality": config.modality, "channels": config.channel_list,
                "sample_rate": config.target_sfreq, "epoch_seconds": config.epoch_duration,
                "label_format": label_format, "subjects": []}
    for label_file in label_files:
        subject = label_file.stem
        labels = np.atleast_1d(np.loadtxt(label_file))
        if label_format == "five":
            labels = map_5to3(labels)
        else:
            if labels.ndim != 1 or not np.isin(labels, [-1, 0, 1, 2]).all():
                raise ValueError(f"{subject}: three-class labels must be -1, 0, 1 or 2")
            labels = labels.astype(np.int64)
        signals, lengths, missing = {}, [len(labels)], []
        if config.modality in ("PSG", "PSG_ACC"):
            values, count, absent = read_channels(input_dir / f"{subject}.edf",
                                                 PSG_CHANNELS, config, allow_missing)
            signals.update(values)
            lengths.append(count)
            missing.extend(absent)
        if config.modality in ("ACC", "PSG_ACC"):
            values, count, absent = read_channels(input_dir / f"{subject}_ACC.edf",
                                                 ACC_CHANNELS, config, allow_missing)
            signals.update(values)
            lengths.append(count)
            missing.extend(absent)
        n_aligned = min(lengths)
        valid = labels[:n_aligned] >= 0
        if not valid.any():
            raise ValueError(f"{subject}: no valid, complete epochs")
        y = labels[:n_aligned][valid]
        np.save(output / "labels" / f"{subject}.npy", y, allow_pickle=False)
        length = int(config.target_sfreq * config.epoch_duration)
        for channel in config.channel_list:
            if channel in signals:
                x = signals[channel][:n_aligned][valid]
            else:
                x = np.zeros((len(y), length), dtype=np.float32)
            np.save(output / "signals" / f"{subject}__{channel}.npy", x, allow_pickle=False)
        metadata["subjects"].append({"id": subject, "epochs": len(y),
                                     "missing_channels": missing,
                                     "original_epoch_indices": np.flatnonzero(valid).tolist()})
        print(f"{subject}: {len(y)} valid epochs, {config.num_channels} channels")
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "aligned")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--modality", type=canonical_modality, default="PSG")
    parser.add_argument("--label-format", choices=["five", "three"], default="five")
    parser.add_argument("--allow-missing-channels", action="store_true")
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.data_root, Config(args.modality),
                    args.label_format, args.allow_missing_channels)


if __name__ == "__main__":
    main()
