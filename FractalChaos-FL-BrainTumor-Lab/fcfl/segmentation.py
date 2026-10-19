"""U-Net tumour localiser."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from .config import DEVICE, UNET_EPOCHS


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1),
        nn.GroupNorm(4, o),
        nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1),
        nn.GroupNorm(4, o),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):

    def __init__(self, w=16):
        super().__init__()
        self.e1, self.e2, self.e3 = (_block(1, w), _block(w, 2 * w), _block(2 * w, 4 * w))
        self.b = _block(4 * w, 8 * w)
        self.u3, self.d3 = (nn.ConvTranspose2d(8 * w, 4 * w, 2, 2), _block(8 * w, 4 * w))
        self.u2, self.d2 = (nn.ConvTranspose2d(4 * w, 2 * w, 2, 2), _block(4 * w, 2 * w))
        self.u1, self.d1 = (nn.ConvTranspose2d(2 * w, w, 2, 2), _block(2 * w, w))
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.b(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(b), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def train_unet(Xi, Mi, epochs=UNET_EPOCHS, batch=16, lr=0.001, seed=0):
    """Train the U-Net localiser on images and radiologist masks."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = TinyUNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(Xi, dtype=torch.float32)
    Mt = torch.tensor(Mi[:, None], dtype=torch.float32)
    for _ in range(epochs):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i : i + batch]
            xb, mb = Xt[idx].to(DEVICE), Mt[idx].to(DEVICE)
            if rng.random() < 0.5:
                xb, mb = xb.flip(-1), mb.flip(-1)
            lo = net(xb)
            p = torch.sigmoid(lo)
            dice = 1 - (2 * (p * mb).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + mb.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(lo, mb) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def predict_prob(net, Xi, batch=64):
    """Tumour probability maps."""
    net.eval()
    out = []
    for i in range(0, len(Xi), batch):
        out.append(
            torch.sigmoid(net(torch.tensor(Xi[i : i + batch], dtype=torch.float32).to(DEVICE)))
            .cpu()
            .numpy()[:, 0]
        )
    return np.concatenate(out)


def _largest(m):
    lbl, n = ndimage.label(m)
    if n < 1:
        return None
    s = np.bincount(lbl.ravel())
    s[0] = 0
    return (lbl == s.argmax()).astype(np.uint8)


def localise(prob, brain, thr=0.5, min_px=16):
    """Largest connected component of the thresholded probability map inside the head."""
    m = _largest((prob > thr) & (brain > 0))
    if m is None or m.sum() < min_px:
        if prob.max() < 0.1:
            return None
        m = _largest((prob > 0.5 * prob.max()) & (brain > 0))
        if m is None or m.sum() < min_px:
            return None
    return m


def dice(a, b):
    """Dice coefficient."""
    a, b = a.astype(bool), b.astype(bool)
    return 2.0 * (a & b).sum() / max(a.sum() + b.sum(), 1)
