"""Simplified SimMIM for the patch embedding and first Swin stage."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import swin_t

from .data import UnlabeledImages


class EarlyStageSimMIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(96))
        self.decoder = nn.Linear(96, 4 * 4 * 3)

    def forward(self, images, mask_ratio=0.5):
        tokens = self.backbone.features[0](images)  # B,32,32,96
        b, h, w, c = tokens.shape
        mask = torch.rand(b, h, w, device=images.device) < mask_ratio
        visible = torch.where(mask[..., None], self.mask_token, tokens)
        predicted = self.decoder(self.backbone.features[1](visible))
        patches = F.unfold(images, kernel_size=4, stride=4).transpose(1, 2).reshape(b, h, w, 48)
        return (predicted - patches).abs()[mask].mean()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optional masked-patch pretraining; no download occurs")
    parser.add_argument("--images", required=True, help="Folder of unlabeled glyph images")
    parser.add_argument("--output", default="checkpoints/simmim.pt")
    parser.add_argument("--imagenet-weights", help="Optional local torchvision Swin-T state_dict")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarlyStageSimMIM()
    if args.imagenet_weights:
        state = torch.load(args.imagenet_weights, map_location="cpu", weights_only=True)
        model.backbone.load_state_dict(state, strict=False)
    model.to(device)
    loader = DataLoader(UnlabeledImages(args.images), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(images)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(images)
        scheduler.step()
        print(f"epoch {epoch + 1}/{args.epochs} masked L1={total / len(loader.dataset):.4f}")
    early = {k: v.cpu() for k, v in model.backbone.state_dict().items() if k.startswith(("features.0.", "features.1."))}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"early_state": early}, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
