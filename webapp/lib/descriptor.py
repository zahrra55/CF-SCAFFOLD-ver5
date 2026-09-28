"""
Fractal tumor-to-brain descriptor.

This module contains the scientific core of Part I of the thesis: the
computation of C = Ft / Fb, the ratio between the box-counting fractal
dimension of the tumor region and that of the same slice's brain tissue with
the tumor removed.

It is kept separate from the interface so that the method can be read, checked
and explained without any reference to how it is displayed.

Every function here is the same code used to produce the results reported in
the thesis; nothing is simplified for demonstration.
"""

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Step 1: separate the three regions
# ---------------------------------------------------------------------------

def brain_mask(gray):
    """Estimate the brain region and exclude the black background.

    The background must be removed before any box-counting estimate, because
    empty boxes would otherwise contribute to the count and bias the dimension
    downward by an amount that depends on how much of the field of view the
    head occupies.

    Steps: threshold, fill interior holes, remove speckle by opening, then keep
    only the largest connected component (the head).

    Parameters
    ----------
    gray : 2-D array, the slice in the range [0, 255]

    Returns
    -------
    2-D uint8 array, 1 inside the brain and 0 outside
    """
    threshold = max(gray.mean() * 0.25, 10)
    mask = gray > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)

    labelled, n_components = ndimage.label(mask)
    if n_components >= 1:
        sizes = np.bincount(labelled.ravel())
        sizes[0] = 0                      # ignore the background label
        mask = labelled == sizes.argmax()  # keep the largest component
    return mask.astype(np.uint8)


def separate_regions(gray, tumor_mask):
    """Return (tumor, tumor-free brain) as two disjoint binary masks.

    The tumor mask supplied with the dataset is intersected with the estimated
    brain so that no mask pixel lying outside the head contributes. The
    tumor-free brain is then the brain with the tumor removed.
    """
    brain = brain_mask(gray)
    tumor = (tumor_mask > 0) & (brain > 0)
    brain_without_tumor = (brain > 0) & (~tumor)
    return tumor.astype(np.uint8), brain_without_tumor.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 2: box-counting fractal dimension
# ---------------------------------------------------------------------------

def box_counting_dimension(gray, region, n_sizes=8):
    """Differential box-counting dimension of `gray` restricted to `region`.

    Intensities are rescaled to [0, 255] *within the region*, so the estimate
    describes the structure of that region rather than its brightness relative
    to the rest of the slice. Without this, an identical tumor in a darker
    slice would receive a different dimension.

    For each box side s, the number of intensity levels spanned inside every
    box that intersects the region is counted. The dimension is the slope of
    log N(s) against log(1/s).

    Returns
    -------
    float, or nan if the region is too small for the estimate to be meaningful
    """
    values = gray[region > 0]
    if values.size < 64:
        return np.nan

    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:            # a uniform region has no structure to measure
        return np.nan

    scaled = np.zeros_like(gray, dtype=np.float64)
    scaled[region > 0] = (gray[region > 0] - lo) / (hi - lo) * 255.0

    height, width = scaled.shape
    max_power = int(np.log2(min(height, width)))
    sizes = [2 ** k for k in range(1, max_power)][:n_sizes]

    counts = []
    for s in sizes:
        boxes = 0
        for i in range(0, height - height % s, s):
            for j in range(0, width - width % s, s):
                box_region = region[i:i + s, j:j + s]
                if box_region.sum() == 0:
                    continue
                block = scaled[i:i + s, j:j + s][box_region > 0]
                if block.size == 0:
                    continue
                # number of intensity levels of height s spanned by this box
                boxes += int(np.floor(block.max() / s)
                             - np.floor(block.min() / s) + 1)
        counts.append(max(boxes, 1))

    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    slope = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)[0]
    return float(slope), sizes, counts


# ---------------------------------------------------------------------------
# Step 3: the descriptor
# ---------------------------------------------------------------------------

def tumor_to_brain_ratio(gray, tumor_mask):
    """Compute C = Ft / Fb together with the intermediate quantities.

    Returns a dictionary so that the interface can display not only the final
    value but every step that produced it, which is what makes the descriptor
    inspectable rather than a black box.

    Keys returned:
        C            the descriptor
        Ft, Fb       the two dimensions
        tumor, brain the two region masks
        curves       the (sizes, counts) pairs used for each fit
        valid        False when the regions are too small to measure
        message      explanation when valid is False
    """
    tumor, brain_free = separate_regions(gray, tumor_mask)

    if tumor.sum() < 16:
        return {"valid": False,
                "message": "The tumor region is smaller than 16 pixels. "
                           "A box-counting estimate over a region this small "
                           "is not meaningful, so the slice is rejected."}
    if brain_free.sum() < 64:
        return {"valid": False,
                "message": "The tumor-free brain region is smaller than 64 "
                           "pixels. There is not enough surrounding tissue to "
                           "form a reference, so the slice is rejected."}

    result_t = box_counting_dimension(gray, tumor)
    result_b = box_counting_dimension(gray, brain_free)

    if not isinstance(result_t, tuple) or not isinstance(result_b, tuple):
        return {"valid": False,
                "message": "One of the two regions is uniform in intensity, "
                           "so no dimension can be estimated from it."}

    Ft, sizes_t, counts_t = result_t
    Fb, sizes_b, counts_b = result_b

    if not (np.isfinite(Ft) and np.isfinite(Fb)) or Fb < 1e-6:
        return {"valid": False,
                "message": "The dimension estimate did not converge on this "
                           "slice."}

    return {"valid": True,
            "C": Ft / Fb,
            "Ft": Ft,
            "Fb": Fb,
            "tumor": tumor,
            "brain": brain_free,
            "curves": {"tumor": (sizes_t, counts_t),
                       "brain": (sizes_b, counts_b)}}


# ---------------------------------------------------------------------------
# Reference values measured on the thesis dataset, for interpretation
# ---------------------------------------------------------------------------

# One-vs-rest AUC of C used alone as a single-feature classifier, measured on
# the 2740 usable Figshare slices. Reported here so the interface can state
# honestly how much weight a single value of C deserves.
DESCRIPTOR_AUC = {
    "Pituitary": 0.907,
    "Glioma": 0.735,
    "Meningioma": 0.613,
}

# Observed range of C on the same slices.
C_RANGE = (0.428, 1.076)
