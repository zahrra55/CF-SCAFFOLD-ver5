"""Reading the Figshare and BraTS data."""

import csv
import glob
import os
import re

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage

from .config import IMG_SIZE, INPUT_ROOT


def _read_pid(cj):
    if "PID" not in cj:
        return None
    try:
        s = "".join((chr(int(c)) for c in np.array(cj["PID"]).ravel() if int(c) > 0))
        return s.strip() or None
    except Exception:
        return None


def read_cjdata(path):
    """Read one Figshare .mat file: image, label (1-3), tumour mask and patient ID."""
    try:
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).ravel()[0])
            image = np.array(cj["image"]).astype(np.float32)
            if image.ndim == 2:
                image = image.T
            mask = None
            if "tumorMask" in cj:
                mask = np.array(cj["tumorMask"]).astype(np.uint8)
                if mask.ndim == 2:
                    mask = mask.T
            return (image, label, mask, _read_pid(cj))
    except Exception:
        return (None, None, None, None)


def resize_pair(img, mask, size=IMG_SIZE):
    """Rescale the image to 0-255 and resize image and mask to size x size."""
    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    g = np.asarray(im).astype(np.float32)
    mk = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (g, (np.asarray(mk) > 127).astype(np.uint8))


def brain_mask_of(gray):
    """Foreground mask of a normalised slice. On Figshare (not skull-stripped) it covers the head."""
    m = gray > max(gray.mean() * 0.25, 10)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, iterations=2)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel())
        s[0] = 0
        m = lbl == s.argmax()
    return m.astype(np.uint8)


def brain_mask_raw(raw, size=IMG_SIZE):
    """Brain mask from raw intensities, for skull-stripped BraTS slices."""
    m = ndimage.binary_fill_holes(raw > 0)
    lbl, n = ndimage.label(m)
    if n >= 1:
        s = np.bincount(lbl.ravel())
        s[0] = 0
        m = lbl == s.argmax()
    mk = Image.fromarray((m * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
    return (np.asarray(mk) > 127).astype(np.uint8)


def brats_grades():
    """Grade (HGG or LGG) of each BraTS subject, from name_mapping.csv or folder names."""
    grade = {}
    for csvp in glob.glob(f"{INPUT_ROOT}/**/name_mapping*.csv", recursive=True):
        with open(csvp) as fh:
            for r in csv.DictReader(fh):
                g = (r.get("Grade") or "").strip().upper()
                for k, v in r.items():
                    if v and "Training" in v and (g in ("HGG", "LGG")):
                        grade[v.strip()] = g
    return grade


def seg_file(d):
    """Segmentation file of a BraTS case, matched case-insensitively."""
    c = [f for f in os.listdir(d) if re.search("seg", f, re.I) and f.lower().endswith((".nii", ".nii.gz"))]
    return os.path.join(d, sorted(c)[0]) if c else None


def find_mat(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".mat"):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def read_pid(cj):
    if "PID" not in cj:
        return None
    raw = np.array(cj["PID"]).ravel()
    try:
        s = "".join((chr(int(c)) for c in raw if int(c) > 0))
        return s.strip() or None
    except Exception:
        return None


def resize_img(img, size):
    from PIL import Image

    im = Image.fromarray(((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)).resize(
        (size, size), Image.BILINEAR
    )
    return np.asarray(im).astype(np.float32)


def load_slices_cnn(output_dir):
    """Images, labels and patient IDs for the image-based federated experiments."""
    cache = os.path.join(output_dir, "cnn_slices.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["Y"], d["PID"]
    X, Y, PID = [], [], []
    for path in find_mat(INPUT_ROOT):
        try:
            with h5py.File(path, "r") as f:
                cj = f["cjdata"]
                lab = int(np.array(cj["label"]).ravel()[0])
                img = np.array(cj["image"]).astype(np.float32)
                if img.ndim == 2:
                    img = img.T
                pid = read_pid(cj)
        except Exception:
            continue
        if pid is None:
            continue
        X.append((resize_img(img, IMG_SIZE) / 255.0).astype(np.float32))
        Y.append(lab - 1)
        PID.append(pid)
    X, Y, PID = np.array(X)[:, None], np.array(Y), np.array(PID)
    np.savez_compressed(cache, X=X, Y=Y, PID=PID)
    return X, Y, PID
