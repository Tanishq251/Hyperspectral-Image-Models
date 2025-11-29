import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


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
