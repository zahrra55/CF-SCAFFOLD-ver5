"""Building and caching the descriptor datasets."""

import glob
import hashlib
import inspect
import os

import nibabel as nib
import numpy as np
from PIL import Image

from .config import BRATS_GLOB, BRATS_MIN_AREA, BRATS_SLICES, FIGSHARE_GLOB, IMG_SIZE, INPUT_ROOT, OUTPUT_DIR
from .descriptors import (
    ALL,
    all_desc,
    boundary_fd,
    boxcount_fd,
    edge_features,
    features_from,
    fuzen2d,
    geometry,
    glcm_region,
    gray_lacunarity,
    lacunarity,
    sampen2d,
    shannon_entropy,
    shape_features,
)
from .io import brain_mask_of, brain_mask_raw, brats_grades, read_cjdata, resize_pair, seg_file


def signature(extra=""):
    """Signature of the descriptor code and parameters, stored with each cache."""
    fns = [
        read_cjdata,
        resize_pair,
        brain_mask_of,
        brain_mask_raw,
        boxcount_fd,
        glcm_region,
        shannon_entropy,
        lacunarity,
        shape_features,
        edge_features,
        features_from,
        sampen2d,
        fuzen2d,
        boundary_fd,
        gray_lacunarity,
        geometry,
        all_desc,
    ]
    src = (
        "".join((inspect.getsource(f) for f in fns))
        + "|".join(ALL)
        + f"|{IMG_SIZE}|{BRATS_SLICES}|{BRATS_MIN_AREA}|"
        + extra
    )
    return hashlib.md5(src.encode()).hexdigest()


def load_cache(path, sig):
    """Load a cache if it was built by the same code; otherwise return None."""
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    if str(d["signature"]) != sig:
        print(f"  cache {os.path.basename(path)} was built by different code: rebuilding")
        return None
    return {k: d[k] for k in d.files}


def save_cache(path, sig, **arrays):
    """Save arrays together with the code signature."""
    np.savez_compressed(path, signature=sig, **arrays)


def build_figshare():
    """Descriptors, geometry, images, masks and head masks of every usable Figshare slice."""
    sig = signature("figshare")
    path = f"{OUTPUT_DIR}/figshare_descriptors.npz"
    d = load_cache(path, sig)
    if d is not None:
        return d
    files = sorted((f for f in glob.glob(FIGSHARE_GLOB, recursive=True) if f.lower().endswith(".mat")))
    if not files:
        raise SystemExit(f"No Figshare .mat files under {INPUT_ROOT}. Attach the dataset.")
    X, Y, P, G, IMG, M, H = [], [], [], [], [], [], []
    skipped = 0
    for k, path_ in enumerate(files):
        img, label, mask, pid = read_cjdata(path_)
        if img is None or pid is None or mask is None:
            skipped += 1
            continue
        g, mk = resize_pair(img, mask)
        head = brain_mask_of(g)
        f = all_desc(g, mk, head)
        if f is None:
            skipped += 1
            continue
        X.append([f[n] for n in ALL])
        Y.append(label - 1)
        P.append(pid)
        G.append(geometry(g, head)[:10])
        IMG.append(g.astype(np.uint8))
        M.append(mk)
        H.append(head)
        if (k + 1) % 500 == 0:
            print(f"  Figshare: {k + 1}/{len(files)}")
    print(f"  Figshare: {len(Y)} slices kept, {skipped} skipped (unreadable or tumour/head region too small)")
    d = dict(
        X=np.array(X),
        Y=np.array(Y),
        P=np.array(P),
        G=np.array(G),
        IMG=np.stack(IMG),
        M=np.stack(M),
        H=np.stack(H),
    )
    save_cache(path, sig, **d)
    return d


def build_brats():
    """Descriptors, sub-region measures and geometry of the five largest tumour slices per BraTS subject."""
    sig = signature("brats")
    path = f"{OUTPUT_DIR}/brats_descriptors.npz"
    d = load_cache(path, sig)
    if d is not None:
        return d
    files = sorted(glob.glob(BRATS_GLOB, recursive=True))
    grade = brats_grades()
    if not files:
        raise SystemExit(f"No BraTS *t1ce.nii files under {INPUT_ROOT}.")
    X, S, G, Y, P = [], [], [], [], []
    no_grade = no_seg = nonstd = 0

    def rs(m):
        return (
            np.asarray(
                Image.fromarray((m * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            )
            > 127
        ).astype(np.uint8)

    for k, t1 in enumerate(files):
        d_ = os.path.dirname(t1)
        sid = os.path.basename(d_)
        g_ = grade.get(sid) or ("HGG" if "/HGG/" in t1 else "LGG" if "/LGG/" in t1 else None)
        if g_ is None:
            no_grade += 1
            continue
        sf = seg_file(d_)
        if sf is None:
            no_seg += 1
            continue
        nonstd += "seg" not in os.path.basename(sf)
        vol = nib.load(t1).get_fdata().astype(np.float32)
        lab = nib.load(sf).get_fdata()
        area = (lab > 0).sum(axis=(0, 1))
        zs = [z for z in np.argsort(area)[::-1][:BRATS_SLICES] if area[z] >= BRATS_MIN_AREA]
        for z in zs:
            g, mk = resize_pair(vol[:, :, z], (lab[:, :, z] > 0).astype(np.uint8))
            br = brain_mask_raw(vol[:, :, z])
            f = all_desc(g, mk, br)
            if f is None:
                continue
            sub, flags = [], []
            for lv in (4, 1):
                reg = rs((lab[:, :, z] == lv).astype(np.uint8)) & br
                present = reg.sum() >= 64
                fd = boxcount_fd(g, reg) if present else 0.0
                gl = gray_lacunarity(g, reg)[0] if present else 0.0
                sub += [fd, gl]
                flags.append(float(present))
            X.append([f[n] for n in ALL])
            S.append(sub + flags)
            G.append(geometry(g, br)[:10])
            Y.append(1 if g_ == "HGG" else 0)
            P.append(sid)
        if (k + 1) % 50 == 0:
            print(f"  BraTS: {k + 1}/{len(files)}")
    print(
        f"  BraTS: {len(set(P))} subjects used; skipped {no_grade} without grade (validation cases), {no_seg} without segmentation; {nonstd} segmentation file(s) with a non-standard name recovered"
    )
    d = dict(X=np.array(X), S=np.array(S, dtype=float), G=np.array(G), Y=np.array(Y), P=np.array(P))
    save_cache(path, sig, **d)
    return d


def unet_features(D, need):
    """Train one U-Net per patient-grouped fold, localise the tumour on the fold's test slices and compute the
    descriptors from the automatic masks. Returns the descriptor matrix, a usability flag and the Dice of every
    slice (0 where localisation failed). The result is cached."""
    import torch
    from sklearn.model_selection import GroupKFold
    from .config import N_FOLDS, UNET_EPOCHS
    from .segmentation import dice, localise, predict_prob, train_unet

    sig = signature("unet|" + "|".join(sorted(need)) + f"|{UNET_EPOCHS}")
    path = f"{OUTPUT_DIR}/unet_features.npz"
    cached = load_cache(path, sig)
    if cached is not None:
        return cached["XA"], cached["okA"], cached["dice_all"]
    X, Y, P = D["X"], D["Y"], D["P"]
    XA = np.full((len(Y), len(ALL)), np.nan)
    okA = np.zeros(len(Y), bool)
    dice_all = np.zeros(len(Y))
    Xi = (D["IMG"].astype(np.float32) / 255.0)[:, None]
    for k, (tr, te) in enumerate(GroupKFold(N_FOLDS).split(X, Y, P)):
        net = train_unet(Xi[tr], D["M"][tr], seed=k)
        prob = predict_prob(net, Xi[te])
        for j, i in enumerate(te):
            loc = localise(prob[j], D["H"][i])
            if loc is None:
                continue
            dice_all[i] = dice(loc, D["M"][i])
            f = all_desc(D["IMG"][i].astype(np.float32), loc, D["H"][i], need=set(need))
            if f is not None:
                XA[i] = [f[n] for n in ALL]
                okA[i] = True
        print(f"  U-Net fold {k + 1}/{N_FOLDS} done")
        del net
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    save_cache(path, sig, XA=XA, okA=okA, dice_all=dice_all)
    return XA, okA, dice_all
