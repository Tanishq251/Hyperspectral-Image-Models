import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.ndimage import label as scipy_label


def split_data(*args, labels, random_state=42):
    """Split data by ratio (train/test or train/val/test)"""
    indices = list(range(len(labels)))
    
    if len(args) == 2:
        train_ratio, test_ratio = args
        if not np.isclose(train_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0. Got {train_ratio + test_ratio}")
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=random_state, stratify=labels
        )
        print(f"Split: Train={train_ratio*100:.0f}% | Test={test_ratio*100:.0f}%")
        return train_idx, test_idx
    
    elif len(args) == 3:
        train_ratio, val_ratio, test_ratio = args
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}")
        
        train_idx, temp_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), random_state=random_state, stratify=labels
        )
        
        temp_labels = [labels[i] for i in temp_idx]
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=(1 - val_size_adjusted), random_state=random_state, stratify=temp_labels
        )
        print(f"Split: Train={train_ratio*100:.0f}% | Val={val_ratio*100:.0f}% | Test={test_ratio*100:.0f}%")
        return train_idx, val_idx, test_idx
    
    else:
        raise ValueError(f"Expected 2 or 3 split ratios, got {len(args)}")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: spatially cut a single component mask into N parts
# ─────────────────────────────────────────────────────────────────────────────
def _spatial_cut(comp_mask, ratios, out_masks):
    """
    Cut `comp_mask` spatially along the longer axis according to `ratios`.
    Pixels are written into the corresponding array in `out_masks`.

    Parameters
    ----------
    comp_mask : bool array (H, W)
    ratios    : list of floats summing to ~1.0, one per output split
    out_masks : list of bool arrays (same shape), one per split
    """
    coords = np.argwhere(comp_mask)
    rows, cols = coords[:, 0], coords[:, 1]
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    H = row_max - row_min + 1
    W = col_max - col_min + 1
    n = len(ratios)
    cum = np.cumsum(ratios)

    if W >= H:                                      # cut along columns
        raw = [col_min + int(round(W * cum[i])) for i in range(n - 1)]
        bounds = [col_min]
        for v in raw:
            bounds.append(max(bounds[-1] + 1, min(v, col_max)))
        bounds.append(col_max + 1)
        for r, c in coords:
            for part in range(n):
                if bounds[part] <= c < bounds[part + 1]:
                    out_masks[part][r, c] = True
                    break
    else:                                           # cut along rows
        raw = [row_min + int(round(H * cum[i])) for i in range(n - 1)]
        bounds = [row_min]
        for v in raw:
            bounds.append(max(bounds[-1] + 1, min(v, row_max)))
        bounds.append(row_max + 1)
        for r, c in coords:
            for part in range(n):
                if bounds[part] <= r < bounds[part + 1]:
                    out_masks[part][r, c] = True
                    break


# ─────────────────────────────────────────────────────────────────────────────
#  CORE: component-level spatially disjoint mask generation
#
#  Rules:
#   • 1 component  → spatially cut into all N splits using target ratios
#   • 2 components → assign smaller wholesale to best-matching split;
#                    spatially cut larger into the remaining split(s)
#   • 3+ components → greedy whole-component assignment (largest-first)
#                     + post-correction: cap train at target_ratio + 5%
#                       by spatially cutting the dominant train component
# ─────────────────────────────────────────────────────────────────────────────
def _generate_disjoint_masks(gt, train_ratio=0.5, val_ratio=0.3):
    """
    Returns (train_mask, val_mask, test_mask) bool arrays of shape (H, W).

    Parameters
    ----------
    gt          : np.ndarray (H, W) – label map, 0 = background
    train_ratio : target fraction for training   (default 0.5)
    val_ratio   : target fraction for validation (default 0.3)
    """
    train_mask = np.zeros_like(gt, dtype=bool)
    val_mask   = np.zeros_like(gt, dtype=bool)
    test_mask  = np.zeros_like(gt, dtype=bool)

    test_ratio = 1.0 - train_ratio - val_ratio
    splits        = ['train', 'val', 'test']
    target_ratios = {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    split_masks   = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    classes = sorted(int(c) for c in np.unique(gt) if c != 0)

    for cls in classes:
        cls_mask = (gt == cls)
        labeled_array, num_components = scipy_label(cls_mask)

        if num_components == 0:
            continue

        # Gather component info
        components = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_array == comp_id)
            n_pixels  = int(comp_mask.sum())
            if n_pixels > 0:
                components.append({'id': comp_id, 'mask': comp_mask, 'size': n_pixels})

        n_comps = len(components)

        # ── CASE 1: 1 component → spatially cut into all 3 splits ──────
        if n_comps == 1:
            ratios  = [target_ratios[s] for s in splits]
            total_r = sum(ratios)
            ratios  = [r / total_r for r in ratios]
            _spatial_cut(components[0]['mask'], ratios,
                         [split_masks[s] for s in splits])
            continue

        # ── CASE 2: 2 components → small wholesale, large spatially cut ──
        if n_comps == 2:
            small, large = sorted(components, key=lambda c: c['size'])
            total_pixels = small['size'] + large['size']
            small_share  = small['size'] / total_pixels

            best_sp   = min(splits, key=lambda s: abs(target_ratios[s] - small_share))
            remaining = [s for s in splits if s != best_sp]

            split_masks[best_sp][small['mask']] = True

            rem_sum    = sum(target_ratios[s] for s in remaining)
            rem_ratios = ([target_ratios[s] / rem_sum for s in remaining]
                          if rem_sum > 0
                          else [1.0 / len(remaining)] * len(remaining))

            if len(remaining) == 1:
                split_masks[remaining[0]][large['mask']] = True
            else:
                _spatial_cut(large['mask'], rem_ratios,
                             [split_masks[s] for s in remaining])
            continue

        # ── CASE 3: 3+ components → greedy whole-component assignment ──
        components.sort(key=lambda c: c['size'], reverse=True)

        current     = {s: 0 for s in splits}
        assignments = {}

        for comp in components:
            best_sp  = splits[0]
            best_dev = float('inf')

            for sp in splits:
                simulated      = dict(current)
                simulated[sp] += comp['size']
                total_sim      = sum(simulated.values())
                dev = (sum(abs(simulated[s] / total_sim - target_ratios[s])
                           for s in splits)
                       if total_sim > 0 else float('inf'))

                if dev < best_dev:
                    best_dev = dev
                    best_sp  = sp

            assignments[comp['id']] = best_sp
            current[best_sp] += comp['size']

        # Safety: ensure every split has ≥ 1 component
        starved = [sp for sp in splits
                   if not any(a == sp for a in assignments.values())]
        px = {s: sum(c['size'] for c in components if assignments[c['id']] == s)
              for s in splits}

        for sp in starved:
            donor = None
            for candidate in sorted(splits, key=lambda s: px[s], reverse=True):
                if sum(1 for a in assignments.values() if a == candidate) >= 2:
                    donor = candidate
                    break

            if donor is None:
                largest = max(components, key=lambda c: c['size'])
                owner   = assignments[largest['id']]
                t_owner = target_ratios[owner]
                t_sp    = target_ratios[sp]
                denom   = (t_owner + t_sp) if (t_owner + t_sp) > 0 else 2.0
                _spatial_cut(largest['mask'],
                             [t_owner / denom, t_sp / denom],
                             [split_masks[owner], split_masks[sp]])
                assignments[largest['id']] = '__split__'
                px[sp] += largest['size'] // 2
                continue

            donor_comps = [c for c in components if assignments[c['id']] == donor]
            victim      = min(donor_comps, key=lambda c: c['size'])
            assignments[victim['id']] = sp
            px[donor] -= victim['size']
            px[sp]    += victim['size']

        # Post-correction: cap train at target_ratio + 5%
        TRAIN_TOL = 0.05
        total_cls_px    = sum(c['size'] for c in components)
        train_comps_now = [c for c in components
                           if assignments.get(c['id']) == 'train']
        train_px_now    = sum(c['size'] for c in train_comps_now)

        if (total_cls_px > 0
                and (train_px_now / total_cls_px) > (train_ratio + TRAIN_TOL)):
            largest_tr  = max(train_comps_now, key=lambda c: c['size'])
            other_tr_px = train_px_now - largest_tr['size']
            keep_in_tr  = max(0, int(round(train_ratio * total_cls_px)) - other_tr_px)
            give_away   = largest_tr['size'] - keep_in_tr

            if give_away > 1 and keep_in_tr < largest_tr['size']:
                assignments[largest_tr['id']] = '__split__'
                half_excess = give_away // 2
                r_tr = keep_in_tr          / largest_tr['size']
                r_va = half_excess         / largest_tr['size']
                r_te = (give_away - half_excess) / largest_tr['size']
                _spatial_cut(largest_tr['mask'],
                             [r_tr, r_va, r_te],
                             [split_masks['train'], split_masks['val'],
                              split_masks['test']])

        # Apply assignments
        for comp in components:
            sp = assignments[comp['id']]
            if sp != '__split__':
                split_masks[sp][comp['mask']] = True

    return train_mask, val_mask, test_mask


def split_disjoint_data(gt, positions, patch_size,
                        random_seed=42, min_samples_per_class=5):
    """
    Component-Level Spatially Disjoint Split  (fixed 50 / 30 / 20 ratios).

    Strategy
    --------
    Uses the corrected component-level algorithm from disjoint_sample.py:
      • 1 connected component  → spatially cut into train / val / test
      • 2 connected components → smaller component goes wholesale to the
                                 best-matching split; larger is spatially cut
                                 among the remaining splits
      • 3+ components          → greedy whole-component assignment + post-
                                 correction to cap train at ≤ 55 %

    Ratios are fixed at train=50 %, val=30 %, test=20 %.

    Args
    ----
    gt                    : 2-D ground truth array  (0 = background)
    positions             : [(row, col, label), …]  – from HyperspectralDataset
    patch_size            : spatial patch size used by the model
    random_seed           : reproducibility seed
    min_samples_per_class : minimum dataset samples per class per split

    Returns
    -------
    train_idx, val_idx, test_idx  — dataset-level indices
    """
    rng = np.random.default_rng(random_seed)
    rows, cols = gt.shape

    print(f"[Disjoint] Component-level split | Image {rows}×{cols} | "
          f"target train=50% val=30% test=20%")

    unique_classes = sorted(int(c) for c in np.unique(gt) if c != 0)

    # ── STEP 1: Build spatial masks ──────────────────────────────────────
    train_mask, val_mask, test_mask = _generate_disjoint_masks(gt)

    # ── STEP 2: Masks → dataset indices ─────────────────────────────────
    train_idx, val_idx, test_idx = [], [], []
    half = patch_size // 2

    for idx, (i, j, _) in enumerate(positions):
        r, c = i + half, j + half
        if train_mask[r, c]:
            train_idx.append(idx)
        elif val_mask[r, c]:
            val_idx.append(idx)
        elif test_mask[r, c]:
            test_idx.append(idx)

    # Uncovered positions go to test
    covered = set(train_idx + val_idx + test_idx)
    for idx in range(len(positions)):
        if idx not in covered:
            test_idx.append(idx)

    # ── STEP 3: Class-repair guarantee ───────────────────────────────────
    labels_arr = np.array([lbl for _, _, lbl in positions])

    train_set = set(train_idx)
    val_set   = set(val_idx)
    test_set  = set(test_idx)

    repaired_classes = []

    for cls in unique_classes:
        cls_all      = set(np.where(labels_arr == cls)[0].tolist())
        cls_in_train = cls_all & train_set
        cls_in_val   = cls_all & val_set
        cls_in_test  = cls_all & test_set

        need_train = max(0, min_samples_per_class - len(cls_in_train))
        need_test  = max(0, min_samples_per_class - len(cls_in_test))

        if need_train == 0 and need_test == 0:
            continue

        repaired_classes.append(cls)

        # Repair train
        if need_train > 0:
            donors = list(cls_in_test)
            rng.shuffle(donors)
            for idx in donors[:need_train]:
                test_set.discard(idx)
                train_set.add(idx)
            still_need = need_train - len(donors[:need_train])
            if still_need > 0:
                donors_val = list(cls_in_val)
                rng.shuffle(donors_val)
                for idx in donors_val[:still_need]:
                    val_set.discard(idx)
                    train_set.add(idx)

        cls_in_train = cls_all & train_set
        cls_in_val   = cls_all & val_set
        cls_in_test  = cls_all & test_set

        # Repair test
        if need_test > 0:
            donors = list(cls_in_train)
            rng.shuffle(donors)
            safe = donors[min_samples_per_class:]
            for idx in safe[:need_test]:
                train_set.discard(idx)
                test_set.add(idx)
            still_need = need_test - len(safe[:need_test])
            if still_need > 0:
                donors_val = list(cls_in_val)
                rng.shuffle(donors_val)
                for idx in donors_val[:still_need]:
                    val_set.discard(idx)
                    test_set.add(idx)

        # Repair val
        cls_in_val = cls_all & val_set
        need_val   = max(0, min_samples_per_class - len(cls_in_val))
        if need_val > 0:
            cls_in_test = cls_all & test_set
            donors = list(cls_in_test)
            rng.shuffle(donors)
            safe = donors[min_samples_per_class:]
            for idx in safe[:need_val]:
                test_set.discard(idx)
                val_set.add(idx)

    train_idx = sorted(train_set)
    val_idx   = sorted(val_set)
    test_idx  = sorted(test_set)

    if repaired_classes:
        print(f"[Disjoint] ⚠ Repaired {len(repaired_classes)} classes with "
              f"<{min_samples_per_class} samples: {repaired_classes}")

    # ── STEP 4: Summary ──────────────────────────────────────────────────
    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"\n[Disjoint] ═══ Component-Level Split Summary ═══")
    print(f"  Train={len(train_idx)} ({100*len(train_idx)/total:.1f}%)  |  "
          f"Val={len(val_idx)} ({100*len(val_idx)/total:.1f}%)  |  "
          f"Test={len(test_idx)} ({100*len(test_idx)/total:.1f}%)")

    print(f"\n  {'Cls':>4}  {'Total':>7}  {'Train':>7}  {'Val':>7}  "
          f"{'Test':>7}  {'Train%':>7}")
    print(f"  {'─'*50}")
    for cls in unique_classes:
        cls_indices = np.where(labels_arr == cls)[0]
        n_total = len(cls_indices)
        n_train = sum(1 for i in cls_indices if i in train_set)
        n_val   = sum(1 for i in cls_indices if i in val_set)
        n_test  = sum(1 for i in cls_indices if i in test_set)
        pct     = 100 * n_train / n_total if n_total > 0 else 0
        print(f"  {cls:>4}  {n_total:>7}  {n_train:>7}  {n_val:>7}  "
              f"{n_test:>7}  {pct:>6.1f}%")

    overlap = np.sum(train_mask & test_mask) + np.sum(train_mask & val_mask)
    if overlap > 0:
        print(f"\n  ❌ WARNING: {overlap} pixels overlap between splits!")
    else:
        print(f"\n  ✅ Zero spatial overlap between all splits")

    return train_idx, val_idx, test_idx




def split_disjoint_samples(*args, gt, positions, patch_size,
                           random_seed=42):
    """
    Component-Level Disjoint Split with Fixed Samples per Class.

    Combines spatial disjoint splitting with fixed sample counts:
        1. First, does a component-level spatial split (50/50) to get
           spatially disjoint train and test regions.
        2. Then, from the train region, picks exactly N samples per class.
        3. Remaining train-region samples + all test-region samples → test.
        4. If val_samples is specified, picks val_samples from test region.

    This ensures spatial disjoint guarantee while giving you exact
    sample counts per class for training.

    Args
    ----
        *args : int
            If 1 arg: (train_samples_per_class,)
            If 2 args: (train_samples_per_class, val_samples_per_class)
        gt                    : 2D ground truth (0 = background)
        positions             : [(row, col, label), ...] from HyperspectralDataset
        patch_size            : spatial patch size used by the model
        random_seed           : reproducibility seed

    Returns
    -------
        train_idx, [val_idx,] test_idx  — dataset-level indices
    """
    rng = np.random.default_rng(random_seed)
    rows, cols = gt.shape

    # Parse args
    if len(args) == 1:
        train_per_class = args[0]
        val_per_class = 0
    elif len(args) == 2:
        train_per_class, val_per_class = args
    else:
        raise ValueError(f"Expected 1 or 2 sample counts, got {len(args)}")

    use_val = val_per_class > 0

    print(f"[Disjoint-Samples] Component-level split + fixed samples | "
          f"Image {rows}×{cols} | train={train_per_class}/cls"
          + (f" val={val_per_class}/cls" if use_val else ""))

    unique_classes = sorted(int(c) for c in np.unique(gt) if c != 0)

    # ── STEP 1: Spatial disjoint split (50/50 as base) ────────
    # Use the component-level split to get spatially disjoint regions
    train_mask = np.zeros_like(gt, dtype=bool)
    test_mask  = np.zeros_like(gt, dtype=bool)

    for cls in unique_classes:
        cls_mask = (gt == cls)
        labeled_array, num_components = scipy_label(cls_mask)

        if num_components == 0:
            continue

        components = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_array == comp_id)
            n_pixels = int(comp_mask.sum())
            if n_pixels == 0:
                continue
            coords = np.argwhere(comp_mask)
            components.append({
                'id': comp_id, 'mask': comp_mask,
                'size': n_pixels, 'coords': coords,
            })

        if len(components) == 1:
            # Single component → spatial cut at 50%
            comp = components[0]
            coords = comp['coords']
            r, c = coords[:, 0], coords[:, 1]
            row_min, row_max = int(r.min()), int(r.max())
            col_min, col_max = int(c.min()), int(c.max())
            H = row_max - row_min + 1
            W = col_max - col_min + 1

            if W >= H:
                split_col = col_min + max(1, min(int(round(W * 0.5)), W - 1))
                for ri, ci in coords:
                    if ci < split_col:
                        train_mask[ri, ci] = True
                    else:
                        test_mask[ri, ci] = True
            else:
                split_row = row_min + max(1, min(int(round(H * 0.5)), H - 1))
                for ri, ci in coords:
                    if ri < split_row:
                        train_mask[ri, ci] = True
                    else:
                        test_mask[ri, ci] = True
            continue

        # Multiple components → assign whole components
        components.sort(key=lambda x: x['size'], reverse=True)
        current = {'train': 0, 'test': 0}
        assignments = {}

        for comp in components:
            if sum(current.values()) == 0:
                assignments[comp['id']] = 'train'
                current['train'] += comp['size']
                continue

            # Pick whichever split is furthest below 50%
            for sp in ['train', 'test']:
                simulated = dict(current)
                simulated[sp] += comp['size']
                total_sim = sum(simulated.values())

            train_if = current['train'] + comp['size']
            test_if  = current['test']  + comp['size']
            dev_train = abs(train_if / (train_if + current['test']) - 0.5)
            dev_test  = abs(current['train'] / (current['train'] + test_if) - 0.5)

            if dev_train <= dev_test:
                assignments[comp['id']] = 'train'
                current['train'] += comp['size']
            else:
                assignments[comp['id']] = 'test'
                current['test'] += comp['size']

        # Imbalance correction
        train_px = sum(co['size'] for co in components if assignments[co['id']] == 'train')
        test_px  = sum(co['size'] for co in components if assignments[co['id']] == 'test')
        if train_px - test_px > 1000:
            for comp in components:
                if assignments[comp['id']] == 'train':
                    assignments[comp['id']] = 'test'
                elif assignments[comp['id']] == 'test':
                    assignments[comp['id']] = 'train'

        for comp in components:
            if assignments[comp['id']] == 'train':
                train_mask[comp['mask']] = True
            else:
                test_mask[comp['mask']] = True

    # ── STEP 2: Map positions to spatial regions ──────────────
    half = patch_size // 2
    labels_arr = np.array([lbl for _, _, lbl in positions])

    train_region_idx = []  # indices in train spatial region
    test_region_idx  = []  # indices in test spatial region

    for idx, (i, j, _) in enumerate(positions):
        r, c = i + half, j + half
        if train_mask[r, c]:
            train_region_idx.append(idx)
        elif test_mask[r, c]:
            test_region_idx.append(idx)
        else:
            test_region_idx.append(idx)  # uncovered → test

    # ── STEP 3: Subsample train region → fixed N per class ────
    train_idx = []
    leftover_train = []  # train region samples not selected

    for cls in unique_classes:
        cls_in_train_region = [i for i in train_region_idx if labels_arr[i] == cls]
        rng.shuffle(cls_in_train_region)

        if len(cls_in_train_region) < train_per_class:
            print(f"  ⚠ Class {cls}: only {len(cls_in_train_region)} samples in train region "
                  f"(need {train_per_class}) — using all available")
            train_idx.extend(cls_in_train_region)
        else:
            train_idx.extend(cls_in_train_region[:train_per_class])
            leftover_train.extend(cls_in_train_region[train_per_class:])

    # ── STEP 4: Handle val if needed ──────────────────────────
    if use_val:
        val_idx = []
        remaining_test = []

        for cls in unique_classes:
            cls_in_test_region = [i for i in test_region_idx if labels_arr[i] == cls]
            rng.shuffle(cls_in_test_region)

            if len(cls_in_test_region) < val_per_class:
                print(f"  ⚠ Class {cls}: only {len(cls_in_test_region)} samples in test region "
                      f"for val (need {val_per_class}) — using all available")
                val_idx.extend(cls_in_test_region)
            else:
                val_idx.extend(cls_in_test_region[:val_per_class])
                remaining_test.extend(cls_in_test_region[val_per_class:])

        # Test = remaining test region + leftover train region
        test_idx = remaining_test + leftover_train
    else:
        val_idx = []
        test_idx = test_region_idx + leftover_train

    # ── STEP 5: Print summary ─────────────────────────────────
    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"\n[Disjoint-Samples] ═══ Split Summary ═══")
    print(f"  Train={len(train_idx)} ({train_per_class}/cls)  |  "
          f"Val={len(val_idx)}  |  Test={len(test_idx)}")

    print(f"\n  {'Cls':>4}  {'Total':>7}  {'Train':>7}  {'Val':>7}  {'Test':>7}")
    print(f"  {'─'*42}")

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    for cls in unique_classes:
        cls_indices = np.where(labels_arr == cls)[0]
        n_total = len(cls_indices)
        n_train = sum(1 for i in cls_indices if i in train_set)
        n_val   = sum(1 for i in cls_indices if i in val_set)
        n_test  = sum(1 for i in cls_indices if i in test_set)
        print(f"  {cls:>4}  {n_total:>7}  {n_train:>7}  {n_val:>7}  {n_test:>7}")

    print(f"\n  ✅ Spatially disjoint + fixed samples per class")

    if use_val:
        return train_idx, val_idx, test_idx
    return train_idx, test_idx


def split_samples(*args, labels, random_state=42):
    """Split data by fixed samples per class"""
    np.random.seed(random_state)
    
    # Group indices by class
    class_indices = {}
    for idx, label in enumerate(labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Shuffle indices within each class
    for label in class_indices:
        np.random.shuffle(class_indices[label])
    
    if len(args) == 1:
        # Two-way split: train_samples, rest
        train_samples = args[0]
        train_idx, test_idx = [], []
        
        min_samples = min(len(indices) for indices in class_indices.values())
        if train_samples > min_samples:
            raise ValueError(f"Requested {train_samples} samples but smallest class has only {min_samples} samples")
        
        for label, indices in class_indices.items():
            train_idx.extend(indices[:train_samples])
            test_idx.extend(indices[train_samples:])
        
        print(f"Split: Train={train_samples} samples/class | Test=remaining samples")
        print(f"Total: Train={len(train_idx)} | Test={len(test_idx)}")
        return train_idx, test_idx
    
    elif len(args) == 2:
        # Three-way split: train_samples, val_samples, rest
        train_samples, val_samples = args
        train_idx, val_idx, test_idx = [], [], []
        
        min_samples = min(len(indices) for indices in class_indices.values())
        if train_samples + val_samples > min_samples:
            raise ValueError(f"Requested {train_samples + val_samples} samples but smallest class has only {min_samples} samples")
        
        for label, indices in class_indices.items():
            train_idx.extend(indices[:train_samples])
            val_idx.extend(indices[train_samples:train_samples + val_samples])
            test_idx.extend(indices[train_samples + val_samples:])
        
        print(f"Split: Train={train_samples} samples/class | Val={val_samples} samples/class | Test=remaining samples")
        print(f"Total: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
        return train_idx, val_idx, test_idx
    
    else:
        raise ValueError(f"Expected 1 or 2 sample counts, got {len(args)}")


def split_samples_fixed_test(*args, labels, random_state=42, test_seed=0):
    """Split by fixed samples per class, keeping the TEST SET identical across runs.

    Unlike `split_samples`, where the per-class shuffle is driven by the run seed
    (so every run gets a different test set), this reserves the test set once
    using a constant `test_seed`:

      1. Shuffle each class with `test_seed` → the first (train+val) samples form
         the train/val POOL, everything after it is the test set. Since
         `test_seed` never changes, `test_idx` is byte-identical for every run.
      2. Shuffle that pool with `random_state` (the per-run seed) → first
         `train_samples` go to train, next `val_samples` go to val.

    So runs still differ in which pool samples are used for training vs
    validation (and in weight init), but mean ± std is computed against one
    fixed test set.
    """
    class_indices = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    if len(args) == 1:
        train_samples, val_samples = args[0], 0
    elif len(args) == 2:
        train_samples, val_samples = args
    else:
        raise ValueError(f"Expected 1 or 2 sample counts, got {len(args)}")

    pool_size = train_samples + val_samples
    min_samples = min(len(indices) for indices in class_indices.values())
    if pool_size > min_samples:
        raise ValueError(f"Requested {pool_size} samples but smallest class has only {min_samples} samples")

    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(class_indices):
        indices = np.array(class_indices[label])

        # ── Fixed carve-out: same pool / same test set for every run ──
        test_rng = np.random.RandomState(test_seed)
        shuffled = indices[test_rng.permutation(len(indices))]
        pool, class_test = shuffled[:pool_size], shuffled[pool_size:]
        test_idx.extend(class_test.tolist())

        # ── Per-run: how the fixed pool is divided into train / val ──
        run_rng = np.random.RandomState(random_state)
        pool = pool[run_rng.permutation(len(pool))]
        train_idx.extend(pool[:train_samples].tolist())
        val_idx.extend(pool[train_samples:pool_size].tolist())

    print(f"[Fixed-Test] Train={train_samples}/class | Val={val_samples}/class | "
          f"Test=remaining (frozen with test_seed={test_seed})")
    print(f"Total: Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")

    if val_samples == 0:
        return train_idx, test_idx
    return train_idx, val_idx, test_idx


def print_class_stats(labels, train_idx, test_idx, val_idx=None, num_classes=None):
    """Print class distribution statistics"""
    if num_classes is None:
        num_classes = len(set(labels))
    
    def get_distribution(indices):
        subset = [labels[i] for i in indices]
        return Counter(subset)
    
    train_counter = get_distribution(train_idx)
    test_counter = get_distribution(test_idx)
    all_classes = sorted(set(labels))
    
    total_train = sum(train_counter.values())
    total_test = sum(test_counter.values())
    
    if val_idx is not None:
        val_counter = get_distribution(val_idx)
        total_val = sum(val_counter.values())
        total_all = total_train + total_val + total_test
        
        header = f"{'Class':^7} | {'Train':^10} | {'Val':^10} | {'Test':^10} | {'Total':^10}"
        print("\n" + "="*len(header))
        print(header)
        print("-"*len(header))
        
        for class_id in all_classes:
            train_count = train_counter.get(class_id, 0)
            val_count = val_counter.get(class_id, 0)
            test_count = test_counter.get(class_id, 0)
            total_count = train_count + val_count + test_count
            print(f"{class_id:^7} | {train_count:^10} | {val_count:^10} | {test_count:^10} | {total_count:^10}")
        
        print("-"*len(header))
        print(f"{'TOTAL':^7} | {total_train:^10} | {total_val:^10} | {total_test:^10} | {total_all:^10}")
        print("="*len(header) + "\n")
    else:
        total_all = total_train + total_test
        
        header = f"{'Class':^7} | {'Train':^10} | {'Test':^10} | {'Total':^10}"
        print("\n" + "="*len(header))
        print(header)
        print("-"*len(header))
        
        for class_id in all_classes:
            train_count = train_counter.get(class_id, 0)
            test_count = test_counter.get(class_id, 0)
            total_count = train_count + test_count
            print(f"{class_id:^7} | {train_count:^10} | {test_count:^10} | {total_count:^10}")
        
        print("-"*len(header))
        print(f"{'TOTAL':^7} | {total_train:^10} | {total_test:^10} | {total_all:^10}")
        print("="*len(header) + "\n")
