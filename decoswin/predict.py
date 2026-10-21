"""Predict a glyph class from a trained checkpoint and a local image."""

import argparse

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from .data import prepare_image
from .model import DeCoSwin


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict one isolated oracle-bone glyph")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    labels = state["labels"]
    model = DeCoSwin(len(labels), state["num_prototypes"], state["topk"])
    model.load_state_dict(state["model"])
    model.to(device).eval()
    with Image.open(args.image) as image:
        x = TF.to_tensor(prepare_image(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(x).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(args.topk, len(labels)))
    for index, score in zip(indices.tolist(), scores.tolist()):
        print(f"{labels[index]}\t{score:.4f}")


if __name__ == "__main__":
    main()
