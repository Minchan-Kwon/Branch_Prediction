import torch
from torch.utils.data import Dataset, DataLoader

class BranchHistoryEmbedding(Dataset):
    """

    """
    def __init__(self, history_indices, targets, vocab_size):
        self.history_indices = history_indices
        self.targets = targets
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.history_indices)

    def __getitem__(self, idx):
        indices = self.history_indices[idx]
        target = self.targets[idx]

        return torch.LongTensor(indices), torch.FloatTensor([target])

    def __repr__(self):
        return f"BranchHistoryEmbeddingDataset(samples={len(self)}, history_length={self.history_indices.shape[1]})"
    
def create_dataloaders(train_data, val_data, test_data, vocab_size,
                    batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders for training, validation, and testing

    Args:
        train_data: tuple of (train_histories, train_targets)
        val_data: tuple of (val_histories, val_targets)
        test_data: tuple of (test_histories, test_targets)
        vocab_size: Length of a single history element (e.g., 2048)
        batch_size: batch size for training
        num_workers: number of parallel workers (use 0 for Colab)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_histories, train_targets = train_data
    val_histories, val_targets = val_data
    test_histories, test_targets = test_data

    #Create datasets
    train_dataset = BranchHistoryEmbedding(train_histories, train_targets, vocab_size)
    val_dataset = BranchHistoryEmbedding(val_histories, val_targets, vocab_size)
    test_dataset = BranchHistoryEmbedding(test_histories, test_targets, vocab_size)

    print(f"\nCreated datasets:")
    print(f"  Train: {train_dataset}")
    print(f"  Val:   {val_dataset}")
    print(f"  Test:  {test_dataset}")

    #Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"\nCreated dataloaders:")
    print(f"  Train: {len(train_loader)} batches of size {batch_size}")
    print(f"  Val: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader

def split_data_temporal(histories, targets, train_ratio=0.7, val_ratio=0.15,
                        test_ratio=0.15):
    """
    Temporal split - NO shuffling, preserves time order

    Test set contains "future" branches that come later in execution.
    This is more realistic for branch prediction evaluation.

    Args:
        histories: numpy array of history indices
        targets: numpy array of taken/not-taken labels
        train_ratio: fraction for training (default 0.7)
        val_ratio: fraction for validation (default 0.15)
        test_ratio: fraction for testing (default 0.15)

    Returns:
        train_histories, train_targets,
        val_histories, val_targets,
        test_histories, test_targets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    num_samples = len(histories)

    #Calculate split points (Preserve Order)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    #Split
    train_histories = histories[:train_end]
    train_targets = targets[:train_end]

    val_histories = histories[train_end:val_end]
    val_targets = targets[train_end:val_end]

    test_histories = histories[val_end:]
    test_targets = targets[val_end:]

    return (train_histories, train_targets,
            val_histories, val_targets,
            test_histories, test_targets)
