"""
==================================================================
 BRISC STRUCTURE PROBE — run this FIRST on Kaggle (takes seconds)
==================================================================
 The external-validation loader found only 8 files, meaning BRISC's
 actual layout on Kaggle differs from the documented one. This probe
 prints the REAL structure so we can fix the loader precisely.
 It does NOT train anything — it just inspects and reports.
==================================================================
"""
import os, glob
from collections import Counter

ROOT = "/kaggle/input"

print("=" * 70)
print("  BRISC STRUCTURE PROBE")
print("=" * 70)

# 1. top-level directories under /kaggle/input
print("\n[1] Top-level under /kaggle/input:")
for d in sorted(glob.glob(f"{ROOT}/*")):
    print(f"    {d}")

# 2. walk the tree, show directory layout (depth-limited) + file counts per dir
print("\n[2] Directory tree (dirs containing files, with counts by extension):")
dir_info = []
for dp, dirs, files in os.walk(ROOT):
    if not files: continue
    exts = Counter(os.path.splitext(f)[1].lower() for f in files)
    dir_info.append((dp, len(files), dict(exts)))
# sort by path, print first ~60 dirs
for dp, n, exts in sorted(dir_info)[:60]:
    rel = dp.replace(ROOT, "")
    print(f"    {rel or '/'}  ({n} files: {exts})")
if len(dir_info) > 60:
    print(f"    ... and {len(dir_info)-60} more directories")

# 3. total image/mask counts by extension across whole tree
print("\n[3] Global extension counts (whole tree):")
all_ext = Counter()
for dp, _, files in os.walk(ROOT):
    for f in files: all_ext[os.path.splitext(f)[1].lower()] += 1
for ext, c in all_ext.most_common(15):
    print(f"    {ext or '(none)'}: {c}")

# 4. sample filenames from any dir that holds images
print("\n[4] Sample image filenames (to see the naming pattern):")
shown = 0
for dp, _, files in os.walk(ROOT):
    imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if imgs and shown < 8:
        print(f"    in {dp.replace(ROOT,'') or '/'}:")
        for f in sorted(imgs)[:4]:
            print(f"        {f}")
        shown += 1

# 5. look specifically for anything mask/segmentation/class related
print("\n[5] Directories whose names hint at masks / segmentation / classes:")
for dp, _, files in os.walk(ROOT):
    low = dp.lower()
    if any(k in low for k in ["mask", "seg", "glioma", "meningioma", "pituitary",
                              "class", "image", "label", "tumor"]):
        n = len([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])
        if n: print(f"    {dp.replace(ROOT,'')}  ({n} images)")

# 6. check for any CSV / manifest files (BRISC ships a manifest)
print("\n[6] CSV / manifest files (BRISC has a metadata manifest):")
for dp, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".csv", ".json", ".txt")):
            print(f"    {os.path.join(dp,f).replace(ROOT,'')}")

print("\n" + "=" * 70)
print("  DONE. Copy ALL of this output back so the loader can be fixed precisely.")
print("=" * 70)
