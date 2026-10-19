"""Fractal, entropy, texture, shape and geometry descriptors."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label as sklabel, regionprops

from .config import ENT_CANDIDATES, ENT_CANDIDATES_HEAD, ENT_TEMPLATES


def boxcount_fd(gray, region, n_sizes=8, G=256.0):
    """Fractal dimension of the grey-level surface inside a region (differential box counting)."""
    v = gray[region > 0]
    if v.size < 64:
        return np.nan
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return np.nan
    g = np.zeros_like(gray, dtype=np.float64)
    g[region > 0] = (gray[region > 0] - lo) / (hi - lo) * (G - 1)
    H, W = g.shape
    M = min(H, W)
    sizes = [2**k for k in range(1, int(np.log2(M)))][:n_sizes]
    r = region > 0
    Ns = []
    for s in sizes:
        h = s * G / M
        Hc, Wc = H - H % s, W - W % s
        B = g[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s)
        R = r[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s)
        valid = R.any(axis=(1, 3))
        mx = np.where(R, B, -np.inf).max(axis=(1, 3))
        mn = np.where(R, B, np.inf).min(axis=(1, 3))
        Ns.append(max(float((np.floor(mx[valid] / h) - np.floor(mn[valid] / h) + 1).sum()), 1.0))
    return float(np.polyfit(np.log(1.0 / np.array(sizes, float)), np.log(np.array(Ns)), 1)[0])


def glcm_region(gray, region, levels=8):
    """GLCM contrast and correlation computed inside a region only."""
    v = gray[region > 0]
    if v.size < 16:
        return (0.0, 0.0)
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-6:
        return (0.0, 0.0)
    q = np.zeros(gray.shape, dtype=np.uint8)
    q[region > 0] = 1 + np.clip(((gray[region > 0] - lo) / (hi - lo) * levels).astype(int), 0, levels - 1)
    P = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels + 1,
        symmetric=True,
        normed=False,
    ).astype(np.float64)[1:, 1:]
    tot = P.sum(axis=(0, 1), keepdims=True)
    if np.any(tot == 0):
        return (0.0, 0.0)
    P = P / tot
    corr = graycoprops(P, "correlation")
    return (float(graycoprops(P, "contrast").mean()), float(np.nan_to_num(corr).mean()))


def shannon_entropy(vals, bins=32):
    """Shannon entropy of an intensity histogram."""
    c, _ = np.histogram(vals, bins=bins)
    p = c[c > 0] / max(c.sum(), 1)
    return float(-(p * np.log2(p)).sum())


def lacunarity(mask, box=8):
    """Gliding-box lacunarity of a binary mask."""
    H, W = mask.shape
    counts = (
        mask[: H - H % box, : W - W % box]
        .reshape(H // box, box, W // box, box)
        .sum(axis=(1, 3))
        .ravel()
        .astype(float)
    )
    if counts.mean() < 1e-6:
        return 0.0
    return float(1 + counts.var() / counts.mean() ** 2)


def shape_features(mask):
    """Solidity, eccentricity, extent and circularity of the largest component."""
    props = regionprops(sklabel(mask))
    if not props:
        return dict(solidity=0.0, eccentricity=0.0, extent=0.0, circularity=0.0)
    p = max(props, key=lambda r: r.area)
    per = p.perimeter if p.perimeter > 0 else 1.0
    return dict(
        solidity=float(p.solidity),
        eccentricity=float(p.eccentricity),
        extent=float(p.extent),
        circularity=float(4 * np.pi * p.area / per**2),
    )


def edge_features(gray, tumour):
    """Edge density, mean gradient and boundary gradient of the tumour."""
    gx, gy = np.gradient(gray.astype(float))
    grad = np.sqrt(gx**2 + gy**2)
    border = ndimage.binary_dilation(tumour) & ~ndimage.binary_erosion(tumour)
    if tumour.sum() == 0:
        return dict(edge_density=0.0, mean_grad=0.0, border_grad=0.0)
    return dict(
        edge_density=float((grad[tumour > 0] > grad.mean()).mean()),
        mean_grad=float(grad[tumour > 0].mean()),
        border_grad=float(grad[border > 0].mean()) if border.sum() else 0.0,
    )


FEATURE_ORDER = [
    "Ft",
    "Fb",
    "C",
    "GLCM_t",
    "correlation_t",
    "GLCM_b",
    "Dispersion_t",
    "lacun_t",
    "lacun_b",
    "size",
    "solidity",
    "eccentricity",
    "extent",
    "circularity",
    "edge_density",
    "mean_grad",
    "border_grad",
]


def features_from(g, tumour_mask, brain):
    """The 17 base descriptors of one slice, or None if a region is too small."""
    tumour = ((tumour_mask > 0) & (brain > 0)).astype(np.uint8)
    rest = ((brain > 0) & (tumour == 0)).astype(np.uint8)
    if tumour.sum() < 16 or rest.sum() < 64:
        return None
    Ft, Fb = boxcount_fd(g, tumour), boxcount_fd(g, rest)
    if not (np.isfinite(Ft) and np.isfinite(Fb) and (Fb > 1e-6)):
        return None
    ct, rt = glcm_region(g, tumour)
    cb, _ = glcm_region(g, rest)
    f = dict(
        Ft=Ft,
        Fb=Fb,
        C=Ft / Fb,
        GLCM_t=ct,
        correlation_t=rt,
        GLCM_b=cb,
        Dispersion_t=shannon_entropy(g[tumour > 0]),
        lacun_t=lacunarity(tumour),
        lacun_b=lacunarity(rest),
        size=float(tumour.sum()),
    )
    f.update(shape_features(tumour))
    f.update(edge_features(g, tumour))
    return f


def _windows(gray, region, k):
    g = gray.astype(np.float64)
    r_ = region > 0
    W = sliding_window_view(g, (k, k))
    pos = np.argwhere(sliding_window_view(r_, (k, k)).all(axis=(2, 3)))
    return (g, r_, W, pos)


def sampen2d(gray, region, m=1, r_factor=0.2, n_templates=ENT_TEMPLATES, n_candidates=ENT_CANDIDATES, seed=0):
    """Two-dimensional sample entropy inside a region (Silva et al., 2016)."""
    g = gray.astype(np.float64)
    r_ = region > 0
    v = g[r_]
    if v.size < 50 or v.std() < 1e-9:
        return np.nan
    r = r_factor * v.std()
    W1 = sliding_window_view(g, (m + 1, m + 1))
    pos = np.argwhere(sliding_window_view(r_, (m + 1, m + 1)).all(axis=(2, 3)))
    if len(pos) < 20:
        return np.nan
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(pos), min(n_templates, len(pos)), replace=False)
    ci = rng.choice(len(pos), min(n_candidates, len(pos)), replace=False)
    win = W1[pos[:, 0], pos[:, 1]]
    P1 = win.reshape(len(pos), -1)
    Pm = win[:, :m, :m].reshape(len(pos), -1)
    A = B = 0
    for s in range(0, len(ti), 100):
        t = ti[s : s + 100]
        dm = np.abs(Pm[t][:, None, :] - Pm[ci][None, :, :]).max(axis=2)
        d1 = np.maximum(dm, np.abs(P1[t][:, None, :] - P1[ci][None, :, :]).max(axis=2))
        ok = t[:, None] != ci[None, :]
        B += int(((dm <= r) & ok).sum())
        A += int(((d1 <= r) & ok).sum())
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def fuzen2d(
    gray, region, m=2, r_factor=0.2, n=2, n_templates=ENT_TEMPLATES, n_candidates=ENT_CANDIDATES, seed=0
):
    """Two-dimensional fuzzy entropy inside a region (Hilal et al., 2020)."""
    g, r_, W1, pos = _windows(gray, region, m + 1)
    v = g[r_]
    if v.size < 50 or v.std() < 1e-9 or len(pos) < 20:
        return np.nan
    r = r_factor * v.std()
    rng = np.random.default_rng(seed)
    ti = rng.choice(len(pos), min(n_templates, len(pos)), replace=False)
    ci = rng.choice(len(pos), min(n_candidates, len(pos)), replace=False)
    win = W1[pos[:, 0], pos[:, 1]]
    P1 = win.reshape(len(pos), -1)
    P1 = P1 - P1.mean(1, keepdims=True)
    Pm = win[:, :m, :m].reshape(len(pos), -1)
    Pm = Pm - Pm.mean(1, keepdims=True)
    sm = s1 = 0.0
    cnt = 0
    for s in range(0, len(ti), 100):
        t = ti[s : s + 100]
        ok = t[:, None] != ci[None]
        dm = np.abs(Pm[t][:, None] - Pm[ci][None]).max(2)
        d1 = np.abs(P1[t][:, None] - P1[ci][None]).max(2)
        sm += np.exp(-((dm / r) ** n))[ok].sum()
        s1 += np.exp(-((d1 / r) ** n))[ok].sum()
        cnt += int(ok.sum())
    return float(np.log(sm / cnt) - np.log(s1 / cnt))


def boundary_fd(mask, min_sizes=3):
    """Box-counting fractal dimension of the tumour boundary."""
    m = mask > 0
    if m.sum() < 30:
        return np.nan
    b = m & ~ndimage.binary_erosion(m)
    H, W = b.shape
    dmax = max(4, int(np.sqrt(4 * m.sum() / np.pi) / 2))
    sizes = [s for s in (2, 3, 4, 6, 8, 12, 16, 24, 32) if s <= dmax]
    if len(sizes) < min_sizes:
        return np.nan
    N = []
    for s in sizes:
        Hc, Wc = H - H % s, W - W % s
        N.append(b[:Hc, :Wc].reshape(Hc // s, s, Wc // s, s).any(axis=(1, 3)).sum())
    return float(-np.polyfit(np.log(sizes), np.log(N), 1)[0])


def gray_lacunarity(gray, region, sizes=(2, 3, 4, 6, 8), min_boxes=30):
    """Gliding-box lacunarity of intensities inside a region: Lambda(2) and slope."""
    g = gray.astype(np.float64)
    r_ = region > 0
    v = g[r_]
    if v.size < 50 or v.max() <= 0:
        return (np.nan, np.nan)
    g = np.where(r_, g, 0.0)
    I = np.pad(g.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    M = np.pad(r_.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    lam, used = [], []
    for s in sizes:
        S = I[s:, s:] - I[:-s, s:] - I[s:, :-s] + I[:-s, :-s]
        C = M[s:, s:] - M[:-s, s:] - M[s:, :-s] + M[:-s, :-s]
        S = S[C == s * s]
        if len(S) < min_boxes or S.mean() <= 0:
            continue
        lam.append((S**2).mean() / S.mean() ** 2)
        used.append(s)
    if not lam:
        return (np.nan, np.nan)
    l2 = lam[0] if used[0] == 2 else np.nan
    slope = float(np.polyfit(np.log(used), np.log(lam), 1)[0]) if len(used) >= 2 else np.nan
    return (float(l2), slope)


def geometry(g, brain):
    """Geometry of the head or brain mask, used as the shortcut control."""
    H, W = brain.shape
    p = max(regionprops(sklabel(brain)), key=lambda r: r.area)
    y0, x0, y1, x1 = p.bbox
    cy, cx = p.centroid
    c = int(round(cx))
    half = min(c, W - c)
    if half < 4:
        msym = isym = 0.0
    else:
        L = brain[:, c - half : c].astype(float)
        R = brain[:, c : c + half][:, ::-1].astype(float)
        msym = float((L * R).sum() / max(np.maximum(L, R).sum(), 1))
        m_ = L * R > 0
        isym = (
            float(np.corrcoef(g[:, c - half : c][m_], g[:, c : c + half][:, ::-1][m_])[0, 1])
            if m_.sum() > 50
            else 0.0
        )
    rows = np.where(brain.any(axis=1))[0]
    third = rows.min() + 2 * (rows.max() - rows.min()) // 3
    return [
        p.area / (H * W),
        (y1 - y0) / max(x1 - x0, 1),
        p.eccentricity,
        p.solidity,
        p.extent,
        abs(p.orientation),
        cy / H,
        cx / W,
        float(brain[third:].sum() / brain.sum()),
        msym,
        float(np.nan_to_num(isym)),
    ]


NEW = ["SampEn_t", "SampEn_b", "FuzEn_t", "FuzEn_b", "Fbound", "GLac2_t", "GLacSlope_t", "mean_grad_in"]


ALL = FEATURE_ORDER + NEW


BASE19 = FEATURE_ORDER + ["SampEn_t", "SampEn_b"]


FUZ19 = FEATURE_ORDER + ["FuzEn_t", "FuzEn_b"]


BLOCK_A = ["Fbound"]
BLOCK_B = ["GLac2_t", "GLacSlope_t"]
HEAD_SIDE = {"Fb", "C", "lacun_b", "GLCM_b", "SampEn_b", "FuzEn_b"}
NOT_STRICT = {"edge_density", "border_grad", "lacun_t", "mean_grad"}
SUB = ["FD_enh", "GLac_enh", "FD_ncrnet", "GLac_ncrnet", "enh_present", "ncrnet_present"]
FLAGS = ["enh_present", "ncrnet_present"]


def all_desc(g, mk, head, need=None):
    """All descriptors of one slice. need: names to compute; the others are NaN."""
    f = features_from(g, mk, head)
    if f is None:
        return None
    t = ((mk > 0) & (head > 0)).astype(np.uint8)
    r = ((head > 0) & (t == 0)).astype(np.uint8)
    want = lambda n: need is None or n in need
    f["SampEn_t"] = sampen2d(g, t) if want("SampEn_t") else np.nan
    f["SampEn_b"] = sampen2d(g, r, n_candidates=ENT_CANDIDATES_HEAD) if want("SampEn_b") else np.nan
    f["FuzEn_t"] = fuzen2d(g, t) if want("FuzEn_t") else np.nan
    f["FuzEn_b"] = fuzen2d(g, r, n_candidates=ENT_CANDIDATES_HEAD) if want("FuzEn_b") else np.nan
    f["Fbound"] = boundary_fd(t) if want("Fbound") else np.nan
    f["GLac2_t"], f["GLacSlope_t"] = (
        gray_lacunarity(g, t) if want("GLac2_t") or want("GLacSlope_t") else (np.nan, np.nan)
    )
    inner = ndimage.binary_erosion(t, iterations=1)
    gx, gy = np.gradient(g.astype(float))
    grad = np.sqrt(gx**2 + gy**2)
    f["mean_grad_in"] = float(grad[inner].mean()) if inner.sum() >= 4 else np.nan
    return f


def cols(names, pool=ALL):
    return [pool.index(n) for n in names]


def derive_sets(final):
    """Fractal-chaos family, tumour-only and strict tumour-only subsets of a descriptor set."""
    ent = [f for f in final if f.startswith(("FuzEn", "SampEn"))]
    fam = ["Ft", "Fb", "C", "lacun_t", "lacun_b"] + ent + [f for f in final if f in BLOCK_A + BLOCK_B]
    tum = [f for f in final if f not in HEAD_SIDE]
    strict = [f for f in tum if f not in NOT_STRICT] + ["mean_grad_in"]
    return (fam, tum, strict)
