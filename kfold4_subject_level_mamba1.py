"""Train, evaluate and summarize four-fold DairySleepNet experiments."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from args import Config, CLASS_NAMES, ROOT, canonical_modality
from model import DairySleepNet


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(path):
    groups = json.loads(Path(path).read_text())["test_groups"]
    if not isinstance(groups, list) or len(groups) != 4 or any(
            not isinstance(group, list) or not group for group in groups):
        raise ValueError("test_groups must contain exactly four nonempty lists")
    subjects = [subject for group in groups for subject in group]
    if any(not isinstance(s, str) or not s or Path(s).name != s or s in (".", "..")
           for s in subjects):
        raise ValueError("Subject IDs must be nonempty file stems, without directory components")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Each subject must occur in exactly one test group")
    return {i + 1: {"train": [s for s in subjects if s not in group], "test": group}
            for i, group in enumerate(groups)}


def load_subjects(config, data_root, subjects):
    raw = Path(data_root) / "raw" / config.modality
    tf = Path(data_root) / "tf" / config.modality
    manifest = json.loads((tf / "manifest.json").read_text())
    if manifest["modality"] != config.modality or manifest["channels"] != config.channel_list:
        raise ValueError("Feature manifest modality or channel order does not match the model")
    if manifest["feature_shape"] != [config.pad_size, config.dim_model]:
        raise ValueError("Feature dimensions do not match the model")
    records, offset = {}, 0
    for record in manifest["subjects"]:
        subject = record["id"]
        if subject in records or record["start"] != offset or record["end"] <= offset:
            raise ValueError("Invalid or overlapping subject boundaries in feature manifest")
        records[subject] = record
        offset = record["end"]
    missing = set(subjects) - set(records)
    if missing:
        raise ValueError(f"Split subjects missing from feature manifest: {sorted(missing)}")
    channels = [np.load(tf / f"TF_{ch}_mean_std.npy", mmap_mode="r", allow_pickle=False)
                for ch in config.channel_list]
    if any(x.shape != (offset, config.pad_size, config.dim_model) for x in channels):
        raise ValueError("Feature shapes do not match subject boundaries")
    data = {}
    for subject in subjects:
        record = records[subject]
        y = np.load(raw / "labels" / f"{subject}.npy", allow_pickle=False)
        if y.shape != (record["end"] - record["start"],) or not np.isin(y, [0, 1, 2]).all():
            raise ValueError(f"Invalid labels or feature/label alignment for {subject}")
        x = np.stack([ch[record["start"]:record["end"]] for ch in channels], axis=1)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite features for {subject}")
        data[subject] = (x.astype(np.float32), y.astype(np.int64))
    return data, manifest


def compute_metrics(y_true, y_pred):
    labels = [0, 1, 2]
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    return {"acc": float(np.mean(y_true == y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "kappa": kappa if np.isfinite(kappa) else None,
            "cm": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
            "class_names": list(CLASS_NAMES), "n_samples": len(y_true)}


def make_loader(x, y, batch_size, weighted=False):
    sampler = None
    if weighted:
        counts = np.bincount(y, minlength=3).astype(np.float32)
        weights = 1.0 / (np.sqrt(counts) + 1e-6)
        sampler = WeightedRandomSampler(torch.tensor(weights[y], dtype=torch.float32),
                                        len(y), replacement=True)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, sampler=sampler, shuffle=False)


@torch.no_grad()
def evaluate_model(model, loader, device, criterion=None):
    model.eval()
    targets, predictions, probabilities, losses = [], [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite model output")
        if criterion is not None:
            losses.append(criterion(logits, y.to(device)).item())
        targets.append(y.numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    return (np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities),
            float(np.mean(losses)) if losses else None)


def train_fold(fold, config, data_root, split, device, out_dir, evaluate_only=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"fold{fold}_best.pt"
    metadata_path = out_dir / f"fold{fold}_run.json"
    if not evaluate_only and any(out_dir.glob(f"fold{fold}_*")):
        raise FileExistsError(f"Fold {fold} already has outputs in {out_dir}; use a fresh --out-dir")
    if evaluate_only:
        saved = json.loads(metadata_path.read_text())
        if saved["config"]["modality"] != config.modality or saved["split"] != split:
            raise ValueError("Checkpoint modality/split does not match the requested evaluation")
        config = Config(**saved["config"])
    set_seed(config.random_state)
    print(f"\nFold {fold}/4 | DairySleepNet | {config.modality} | {device}", flush=True)
    print(f"Train: {split['train']}\nTest: {split['test']}", flush=True)
    data, manifest = load_subjects(config, data_root, split["train"] + split["test"])
    if evaluate_only and manifest != saved["preprocessing"]:
        raise ValueError("Preprocessing manifest differs from the saved experiment")
    x_test = np.concatenate([data[s][0] for s in split["test"]])
    y_test = np.concatenate([data[s][1] for s in split["test"]])
    test_loader = make_loader(x_test, y_test, config.batch_size)
    model = DairySleepNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if not evaluate_only:
        x = np.concatenate([data[s][0] for s in split["train"]])
        y = np.concatenate([data[s][1] for s in split["train"]])
        if (np.bincount(y, minlength=3) < 2).any():
            raise ValueError("Each class needs at least two training epochs for stratified validation")
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=config.validation_fraction, stratify=y,
                random_state=config.random_state)
        except ValueError as error:
            raise ValueError("Not enough epochs for a three-class 10% validation split") from error
        train_loader = make_loader(x_train, y_train, config.batch_size, weighted=True)
        val_loader = make_loader(x_val, y_val, config.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        save_json(metadata_path, {"config": config.to_dict(), "split": split,
                                 "preprocessing": manifest, "torch_version": str(torch.__version__),
                                 "train_epochs": len(y_train), "validation_epochs": len(y_val),
                                 "test_epochs": len(y_test)})
        history = []
        best, stale = -1.0, 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            losses, targets, predictions = [], [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
                targets.append(yb.cpu().numpy())
                predictions.append(logits.argmax(1).detach().cpu().numpy())
            train_f1 = f1_score(np.concatenate(targets), np.concatenate(predictions),
                                labels=[0, 1, 2], average="macro", zero_division=0)
            yt, yp, _, val_loss = evaluate_model(model, val_loader, device, criterion)
            val_f1 = compute_metrics(yt, yp)["macro_f1"]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "train_f1": float(train_f1), "val_loss": val_loss, "val_f1": val_f1})
            print(f"Epoch {epoch:3d} | Loss {np.mean(losses):.4f} | "
                  f"Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f}", flush=True)
            if val_f1 > best + 1e-4:
                best, stale = val_f1, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                stale += 1
                if stale >= config.patience:
                    break
        save_json(out_dir / f"fold{fold}_history.json", history)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    yt, yp, prob, _ = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(yt, yp)
    save_json(out_dir / f"fold{fold}_metrics.json", metrics)
    for name, values in [("y_true", yt), ("y_pred", yp), ("y_prob", prob)]:
        np.save(out_dir / f"fold{fold}_{name}.npy", values, allow_pickle=False)
    save_json(out_dir / f"fold{fold}_test_subjects.json", [
        {"id": subject, "epochs": len(data[subject][1])} for subject in split["test"]])
    print(f"Fold {fold} Results: ACC={metrics['acc']:.4f}, "
          f"Macro F1={metrics['macro_f1']:.4f}, Kappa={metrics['kappa']}", flush=True)
    return metrics


def summarize_results(out_dir):
    out_dir = Path(out_dir)
    metrics, runs = [], []
    for fold in range(1, 5):
        metrics.append(json.loads((out_dir / f"fold{fold}_metrics.json").read_text()))
        runs.append(json.loads((out_dir / f"fold{fold}_run.json").read_text()))
    if any(run["config"] != runs[0]["config"] or run["preprocessing"] != runs[0]["preprocessing"]
           for run in runs[1:]):
        raise ValueError("Cannot summarize folds with different configurations or preprocessing")
    tests = [s for run in runs for s in run["split"]["test"]]
    if len(tests) != len(set(tests)) or any(
            set(run["split"]["train"]) != set(tests) - set(run["split"]["test"]) for run in runs):
        raise ValueError("Folds do not form a complete, non-overlapping cross-validation partition")
    summary = {"modality": runs[0]["config"]["modality"], "folds": metrics,
               "std_ddof": 0, "mean": {}, "std": {}}
    for key in ("acc", "macro_f1", "weighted_f1", "kappa"):
        values = [m[key] for m in metrics]
        summary["mean"][key] = None if None in values else float(np.mean(values))
        summary["std"][key] = None if None in values else float(np.std(values))
    summary["summed_confusion_matrix"] = np.sum([m["cm"] for m in metrics], axis=0).tolist()
    save_json(out_dir / "summary.json", summary)
    print(json.dumps({"modality": summary["modality"], "mean": summary["mean"],
                      "std": summary["std"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1, 5))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--modality", type=canonical_modality, default="PSG")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--split-file", type=Path, default=ROOT / "configs" / "folds.json")
    parser.add_argument("--out-dir", "--out_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir or ROOT / "outputs" / args.modality
    if args.summary:
        summarize_results(out_dir)
        return
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    config = Config(modality=args.modality, epochs=args.epochs,
                    batch_size=args.batch_size, random_state=args.seed)
    splits = load_splits(args.split_file)
    for fold in ([args.fold] if args.fold else range(1, 5)):
        train_fold(fold, config, args.data_root, splits[fold], device, out_dir, args.evaluate_only)
    if args.fold is None:
        summarize_results(out_dir)


if __name__ == "__main__":
    main()
