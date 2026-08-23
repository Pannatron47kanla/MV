"""
Re-split a YOLO-format dataset into Train/Validation/Test at a 70/15/15 ratio.

Images that were augmented from the same source photo (Roboflow exports them
as "<basename>_jpg.rf.<hash>.jpg") are grouped together first, so all
augmented copies of one source image always land in the same split -- this
avoids leaking near-duplicate images across train/valid/test.

Usage (run from anywhere; --src/--dst default to myDogs2 next to this script):
    python split_dataset.py --train 0.70 --valid 0.15 --test 0.15 --seed 42
"""

import argparse
import random
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GROUP_KEY_RE = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def group_key(stem: str) -> str:
    """Roboflow names augmented copies '<source>_jpg.rf.<hash>'.
    Strip the '.rf.<hash>' suffix so all copies of one source image share a key."""
    return GROUP_KEY_RE.sub("", stem)


def find_pairs(src: Path):
    """Collect (image_path, label_path) pairs from src/{train,valid,test}/{images,labels}
    or, if src has no split subfolders, from src/images + src/labels directly."""
    split_dirs = [d for d in ("train", "valid", "test") if (src / d / "images").is_dir()]
    search_roots = [src / d for d in split_dirs] if split_dirs else [src]

    pairs = []
    for root in search_roots:
        img_dir = root / "images"
        lbl_dir = root / "labels"
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"warning: no label for {img_path}, skipping")
                continue
            pairs.append((img_path, lbl_path))
    return pairs


def _greedy_pack(items, ratios, target):
    """One greedy pass: largest groups first, each placed into whichever
    split is furthest below its target count. Used as a fallback for very
    large inputs where the exact DP below would be too slow."""
    current = {name: 0 for name in ratios}
    assigned = {name: [] for name in ratios}
    for key, files in sorted(items, key=lambda kv: -len(kv[1])):
        split = max(current, key=lambda s: target[s] - current[s])
        assigned[split].append((key, files))
        current[split] += len(files)
    return assigned, current


def _exact_optimal_split(items, ratios, target, total):
    """Exact DP: choose a split (train/valid/test) per group to minimize
    sum(|count - target|) across splits, without ever breaking a group up.
    State = (train_count, valid_count) after i groups; test_count is
    implied by total images seen so far. Feasible because a classroom
    dataset has few enough source-photo groups that the state space
    ((total+1)^2) stays small."""
    names = list(ratios.keys())  # e.g. ["train", "valid", "test"]
    sizes = [len(files) for _, files in items]

    # dp[i]: {(train_count, valid_count): (prev_state, choice)}
    dp = [dict() for _ in range(len(items) + 1)]
    dp[0][(0, 0)] = None
    for i, size in enumerate(sizes):
        for (t, v) in dp[i]:
            for choice in names:
                nt, nv = t, v
                if choice == "train":
                    nt += size
                elif choice == "valid":
                    nv += size
                # "test" leaves (t, v) unchanged; test_count = running_total - t - v
                state = (nt, nv)
                if state not in dp[i + 1]:
                    dp[i + 1][state] = ((t, v), choice)

    best_state, best_score = None, None
    for (t, v) in dp[len(items)]:
        test_count = total - t - v
        counts = {"train": t, "valid": v, "test": test_count}
        score = sum(abs(counts[n] - target[n]) for n in names)
        if best_score is None or score < best_score:
            best_score, best_state = score, (t, v)

    choices = [None] * len(items)
    state = best_state
    for i in range(len(items), 0, -1):
        prev_state, choice = dp[i][state]
        choices[i - 1] = choice
        state = prev_state

    assigned = {name: [] for name in names}
    current = {name: 0 for name in names}
    for (key, files), choice in zip(items, choices):
        assigned[choice].append((key, files))
        current[choice] += len(files)
    return assigned, current


def balanced_group_split(groups: dict, ratios: dict, seed: int, exact_state_limit: int = 200_000):
    """Assign whole groups (never splitting a group across sets) to
    train/valid/test so the resulting image counts land as close as
    possible to the requested ratios. Solved exactly via DP when the
    dataset is small enough (typical classroom size); falls back to a
    greedy bin-covering heuristic for very large datasets where the exact
    DP's state space would be too big. --seed only affects which specific
    images end up in which split among ties, not the resulting split sizes."""
    total = sum(len(v) for v in groups.values())
    target = {name: total * ratio for name, ratio in ratios.items()}

    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)

    if (total + 1) ** 2 <= exact_state_limit:
        assigned, current = _exact_optimal_split(items, ratios, target, total)
    else:
        assigned, current = _greedy_pack(items, ratios, target)

    return assigned, current, target


def write_split(dst: Path, split_name: str, files):
    img_out = dst / split_name / "images"
    lbl_out = dst / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in files:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(lbl_path, lbl_out / lbl_path.name)


def read_classes(src: Path):
    yaml_path = src / "data.yaml"
    if not yaml_path.exists():
        return None, None
    nc, names = None, None
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("nc:"):
            nc = line.split(":", 1)[1].strip()
        elif line.startswith("names:"):
            names = line.split(":", 1)[1].strip()
    return nc, names


def write_data_yaml(dst: Path, nc, names):
    lines = [
        "train: ../train/images",
        "val: ../valid/images",
        "test: ../test/images",
        "",
        f"nc: {nc if nc is not None else '0'}",
        f"names: {names if names is not None else '[]'}",
        "",
    ]
    (dst / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SCRIPT_DIR / "Add24" / "myDogs2")
    ap.add_argument("--dst", type=Path, default=SCRIPT_DIR / "Add24" / "myDogs2_split_70_15_15")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ratios = {"train": args.train, "valid": args.valid, "test": args.test}
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"train+valid+test must sum to 1.0, got {ratio_sum}")

    pairs = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"no image/label pairs found under {args.src}")

    groups = {}
    for img_path, lbl_path in pairs:
        key = group_key(img_path.stem)
        groups.setdefault(key, []).append((img_path, lbl_path))

    assigned, current, target = balanced_group_split(groups, ratios, args.seed)

    if args.dst.exists():
        shutil.rmtree(args.dst)

    for split_name, group_list in assigned.items():
        files = [pair for _, pair_list in group_list for pair in pair_list]
        write_split(args.dst, split_name, files)

    nc, names = read_classes(args.src)
    write_data_yaml(args.dst, nc, names)

    total = sum(current.values())
    print(f"source pairs found     : {len(pairs)}")
    print(f"source groups (photos)  : {len(groups)}")
    print(f"output dataset          : {args.dst}")
    print()
    print(f"{'split':<8}{'groups':>8}{'images':>8}{'target':>8}{'%':>7}")
    for split_name in ("train", "valid", "test"):
        n_groups = len(assigned[split_name])
        n_images = current[split_name]
        pct = 100 * n_images / total
        print(f"{split_name:<8}{n_groups:>8}{n_images:>8}{target[split_name]:>8.1f}{pct:>6.1f}%")


if __name__ == "__main__":
    main()
