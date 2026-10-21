"""Core modules corresponding to equations (1)-(6) of the DeCoSwin paper."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import swin_t


class AOPA(nn.Module):
    """Sample an 8x8 grid, match shared prototypes, and gate stage-2 anchors."""

    def __init__(self, channels=192, num_prototypes=256, topk=16, tau=0.5):
        super().__init__()
        if not 1 <= topk <= num_prototypes:
            raise ValueError("topk must be in [1, num_prototypes]")
        self.topk = topk
        self.tau = tau
        # The paper specifies two offset convolutions but not their hidden width.
        # A narrower hidden layer keeps the model close to its reported size.
        offset_hidden = max(4, channels // 3)
        gate_hidden = max(8, channels * 2 // 3)
        # A stride-2 convolution predicts one offset for each 2x2 feature cell.
        self.offset = nn.Sequential(
            nn.Conv2d(channels, offset_hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(offset_hidden, 2, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, channels) * 0.02)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, gate_hidden, 3, padding=1),
            nn.BatchNorm2d(gate_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, channels, 1),
            nn.Sigmoid(),
        )

    def sampled_features(self, x):
        """Return B,C,H/2,W/2 samples and offsets measured in feature pixels."""
        b, _, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("AOPA needs even spatial dimensions")
        gh, gw = h // 2, w // 2
        # tanh bounds offsets to 1.5 stage-2 pixels; paper leaves range unspecified.
        offsets = self.offset(x).tanh() * 1.5
        yy, xx = torch.meshgrid(
            torch.arange(gh, device=x.device, dtype=x.dtype),
            torch.arange(gw, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        anchor_x, anchor_y = 2 * xx + 0.5, 2 * yy + 0.5
        px = anchor_x[None] + offsets[:, 0]
        py = anchor_y[None] + offsets[:, 1]
        # align_corners=False maps pixel center p to 2*(p+0.5)/size-1.
        grid = torch.stack((2 * (px + 0.5) / w - 1, 2 * (py + 0.5) / h - 1), -1)
        samples = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return samples, offsets

    def forward(self, x):
        samples, _ = self.sampled_features(x)
        b, c, gh, gw = samples.shape
        tokens = samples.flatten(2).transpose(1, 2)  # B,N,C
        score = F.normalize(tokens, dim=-1) @ F.normalize(self.prototypes, dim=-1).T
        score = score / self.tau
        values, indices = score.topk(self.topk, dim=-1)
        attention = values.softmax(dim=-1)
        chosen = self.prototypes[indices]  # B,N,k,C
        response = (attention.unsqueeze(-1) * chosen).sum(dim=-2)
        response = response.transpose(1, 2).reshape(b, c, gh, gw)
        gate = F.interpolate(self.gate(response), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * (1 + gate)

    def orthogonality_loss(self):
        """Mean absolute off-diagonal cosine similarity, as in equation (5)."""
        p = F.normalize(self.prototypes, dim=-1)
        sim = (p @ p.T).abs()
        k = sim.shape[0]
        return (sim.sum() - sim.diagonal().sum()) / (k * (k - 1))


class DeformableDepthwiseConv(nn.Module):
    """Small pure-PyTorch deformable depthwise 3x3 convolution.

    The paper does not publish the exact deformable operator. Each kernel point
    gets a learned 2D offset; grid_sample supplies bilinear interpolation.
    """

    def __init__(self, channels, dilation):
        super().__init__()
        self.dilation = dilation
        self.offset = nn.Conv2d(channels, 18, 3, padding=1)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.weight = nn.Parameter(torch.empty(channels, 9))
        nn.init.normal_(self.weight, std=(1 / 9) ** 0.5)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        offset = self.offset(x).tanh().reshape(b, 9, 2, h, w) * 1.5
        yy, xx = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype),
            torch.arange(w, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        result = torch.zeros_like(x)
        for i, (dy, dx) in enumerate((
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )):
            px = xx[None] + dx * self.dilation + offset[:, i, 0]
            py = yy[None] + dy * self.dilation + offset[:, i, 1]
            grid = torch.stack((2 * (px + 0.5) / w - 1, 2 * (py + 0.5) / h - 1), -1)
            sampled = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
            result = result + sampled * self.weight[:, i][None, :, None, None]
        return self.act(self.norm(self.pointwise(result + self.bias[None, :, None, None])))


class DMF(nn.Module):
    """Three deformable paths with dilation 1/3/5 and a scaled residual."""

    def __init__(self, channels, hidden=96):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.paths = nn.ModuleList(DeformableDepthwiseConv(hidden, d) for d in (1, 3, 5))
        self.fuse = nn.Conv2d(hidden * 3, channels, 1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        z = self.project(x)
        return self.fuse(torch.cat([path(z) for path in self.paths], dim=1)) + self.alpha * x


class DeCoSwin(nn.Module):
    """Swin-T at 128x128 with AOPA, two DMFs, and 384+768+768 fusion."""

    def __init__(self, num_classes, num_prototypes=256, topk=16):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.backbone = swin_t(weights=None)
        self.backbone.head = nn.Identity()
        self.aopa = AOPA(192, num_prototypes, topk)
        self.dmf3 = DMF(384)
        self.dmf4 = DMF(768)
        self.fusion = nn.Sequential(nn.Linear(384 + 768 + 768, 768), nn.LayerNorm(768), nn.ReLU(inplace=True))
        self.classifier = nn.Linear(768, num_classes)

    @staticmethod
    def _nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def stage2(self, x):
        for layer in self.backbone.features[:4]:
            x = layer(x)
        return self._nchw(x)

    def forward(self, x):
        if x.shape[-2:] != (128, 128):
            raise ValueError("This implementation expects 128x128 inputs")
        x = self.stage2(x)  # B,192,16,16
        x = self.aopa(x)
        x = self.backbone.features[4](x.permute(0, 2, 3, 1))
        x = self.backbone.features[5](x)
        stage3 = self.dmf3(self._nchw(x))  # B,384,8,8
        x = self.backbone.features[6](x)
        x = self.backbone.features[7](x)
        final = self.backbone.norm(x)
        stage4 = self.dmf4(self._nchw(x))  # B,768,4,4
        pooled = [F.adaptive_avg_pool2d(y, 1).flatten(1) for y in (stage3, stage4, self._nchw(final))]
        return self.classifier(self.fusion(torch.cat(pooled, dim=1)))

    def orthogonality_loss(self):
        return self.aopa.orthogonality_loss()
