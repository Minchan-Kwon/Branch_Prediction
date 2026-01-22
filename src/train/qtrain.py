import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer

def train_qat_model(model, train_loader, val_loader,
                    groupsize=32,
                    num_epochs=80, learning_rate=0.001,
                    device='cuda', patience=10):
    """
    QAT Training using Int8DynActInt4WeightQATQuantizer
    """

    model = model.to(device)

    # Quantizer 생성 및 prepare
    quantizer = Int8DynActInt4WeightQATQuantizer(groupsize=groupsize)
    model = quantizer.prepare(model)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    script_path = Path(__file__).resolve().parents[2]
    intermediate_path = script_path / 'run' / 'model'
    intermediate_path.mkdir(parents=True, exist_ok=True)
    best_model_path = f'{intermediate_path}/{model.__class__.__name__}_quantized_best_state.pth'

    print(f"\n{'='*70}")
    print(f"QAT TRAINING START (groupsize={groupsize})")
    print(f"{'='*70}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    print(f"Quantization: INT8 Dynamic Activation + INT4 Weight")
    print(f"{'='*70}\n")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_histories, batch_targets in train_loader:
            batch_histories = batch_histories.to(device)
            batch_targets = batch_targets.to(device)

            predictions = model(batch_histories)
            loss = criterion(predictions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (predictions > 0.5).float()
            train_correct += (predicted == batch_targets).sum().item()
            train_total += batch_targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_histories, batch_targets in val_loader:
                batch_histories = batch_histories.to(device)
                batch_targets = batch_targets.to(device)

                predictions = model(batch_histories)
                loss = criterion(predictions, batch_targets)

                val_loss += loss.item()
                predicted = (predictions > 0.5).float()
                val_correct += (predicted == batch_targets).sum().item()
                val_total += batch_targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(current_lr)

        scheduler.step(val_accuracy)

        # Save best
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, best_model_path)
            print(f"Epoch [{epoch+1}/{num_epochs}] Val Acc: {val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"Epoch [{epoch+1}/{num_epochs}] Train: {train_accuracy:.4f}, Val: {val_accuracy:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Convert to quantized model
    print(f"\Quantizing model")
    model = quantizer.convert(model)

    print(f"\nQuantization Complete. Best Val Acc: {best_val_accuracy:.4f}\n")

    return history, best_model_path, model