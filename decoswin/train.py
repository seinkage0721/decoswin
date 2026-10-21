"""Fine-tune and evaluate DeCoSwin on a user-provided CSV manifest."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import ManifestDataset, class_counts, read_manifest, training_labels
from .model import DeCoSwin


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_balanced_weights(counts, beta=0.999):
    """Effective-number weights, rescaled to mean 1 across train classes."""
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError("Each train class must have at least one sample")
    weights = (1 - beta) / (1 - beta ** counts)
    return weights / weights.mean()


@torch.no_grad()
def initialize_prototypes(model, loader, device, max_batches=20, iterations=10):
    """Small K-means initializer on sampled stage-2 training features."""
    model.eval()
    vectors = []
    for batch_index, (images, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        sampled, _ = model.aopa.sampled_features(model.stage2(images.to(device)))
        vectors.append(sampled.flatten(2).transpose(1, 2).reshape(-1, 192).cpu())
    points = torch.cat(vectors, dim=0)
    k = model.aopa.prototypes.shape[0]
    if len(points) < k:
        raise ValueError(f"Need at least {k} sampled patches to initialize prototypes")
    centers = points[torch.randperm(len(points))[:k]].clone()
    for _ in range(iterations):
        group = torch.cdist(points, centers).argmin(dim=1)
        sums = torch.zeros_like(centers).index_add_(0, group, points)
        counts = torch.bincount(group, minlength=k)
        updated = sums / counts.clamp_min(1).unsqueeze(1)
        centers = torch.where(counts[:, None] > 0, updated, centers)
    model.aopa.prototypes.copy_(centers.to(device))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct1 = correct10 = count = 0
    for images, targets in loader:
        scores = model(images.to(device))
        targets = targets.to(device)
        top = scores.topk(min(10, scores.shape[1]), dim=1).indices
        correct1 += (top[:, 0] == targets).sum().item()
        correct10 += (top == targets[:, None]).any(dim=1).sum().item()
        count += len(targets)
    return {"top1": correct1 / count, "top10": correct10 / count, "count": count}


def load_optional_weights(model, imagenet_path=None, simmim_path=None):
    if imagenet_path:
        state = torch.load(imagenet_path, map_location="cpu", weights_only=True)
        # torchvision's published Swin-T checkpoint includes its 1000-class head.
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        if missing or any(k not in ("head.weight", "head.bias") for k in unexpected):
            raise ValueError(f"Unexpected ImageNet checkpoint keys: missing={missing[:5]}, unexpected={unexpected[:5]}")
    if simmim_path:
        state = torch.load(simmim_path, map_location="cpu", weights_only=True)["early_state"]
        model.backbone.load_state_dict(state, strict=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train DeCoSwin")
    parser.add_argument("--manifest", required=True, help="CSV with path,label,split; split=train/val/test")
    parser.add_argument("--output", default="checkpoints/decoswin.pt")
    parser.add_argument("--imagenet-weights", help="Optional local torchvision Swin-T checkpoint")
    parser.add_argument("--simmim-weights", help="Optional output of python -m decoswin.pretrain")
    parser.add_argument("--init-prototypes", action="store_true", help="Initialize dictionary by K-means on training features")
    parser.add_argument("--num-prototypes", type=int, default=256)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    set_seed(args.seed)
    rows = read_manifest(args.manifest)
    labels = training_labels(rows)
    train_set = ManifestDataset(rows, "train", labels, train=True)
    val_set = ManifestDataset(rows, "val", labels) if any(s == "val" for _, _, s in rows) else None
    test_set = ManifestDataset(rows, "test", labels) if any(s == "test" for _, _, s in rows) else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers) if val_set else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers) if test_set else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeCoSwin(len(labels), args.num_prototypes, args.topk)
    load_optional_weights(model, args.imagenet_weights, args.simmim_weights)
    model.to(device)
    if args.init_prototypes:
        # Use unaugmented images so the dictionary starts from actual glyphs.
        init_set = ManifestDataset(rows, "train", labels)
        init_loader = DataLoader(init_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        initialize_prototypes(model, init_loader, device)
    weights = class_balanced_weights(class_counts(train_set, len(labels))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_top1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            # Mean over batch matches Eq. (7), unlike CE's weight-normalized mean.
            classification = (F.cross_entropy(scores, targets, reduction="none") * weights[targets]).mean()
            loss = classification + 1e-3 * model.orthogonality_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)
        scheduler.step()
        metrics = evaluate(model, val_loader, device) if val_loader else None
        # With a validation split save best; otherwise save the final epoch.
        should_save = metrics["top1"] > best_top1 if metrics else True
        if should_save:
            best_top1 = metrics["top1"] if metrics else -1.0
            torch.save({"model": model.state_dict(), "labels": list(labels), "num_prototypes": args.num_prototypes, "topk": args.topk, "epoch": epoch + 1}, output)
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / len(train_set), 5), "val": metrics}, ensure_ascii=False))
    print(f"Saved checkpoint: {output}")
    if test_loader:
        state = torch.load(output, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        print("test:", json.dumps(evaluate(model, test_loader, device), ensure_ascii=False))


if __name__ == "__main__":
    main()
