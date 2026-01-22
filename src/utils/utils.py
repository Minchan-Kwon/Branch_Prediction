import torch
import numpy as np
from matplotlib import pyplot as plt

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot learning rate
    axes[2].plot(history['learning_rate'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved to 'training_history.png'")
    
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set

    Returns:
        accuracy, precision, recall, f1_score
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_histories, batch_targets in test_loader:
            batch_histories = batch_histories.to(device)
            batch_targets = batch_targets.to(device)

            predictions = model(batch_histories)
            predicted = (predictions > 0.5).float()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())

    all_predictions = np.array(all_predictions).flatten()
    all_targets = np.array(all_targets).flatten()

    # Calculate metrics
    accuracy = np.mean(all_predictions == all_targets)

    # Precision, Recall, F1
    true_positives = np.sum((all_predictions == 1) & (all_targets == 1))
    false_positives = np.sum((all_predictions == 1) & (all_targets == 0))
    false_negatives = np.sum((all_predictions == 0) & (all_targets == 1))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Test Set Evaluation")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return accuracy, precision, recall, f1    
    
def get_model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)