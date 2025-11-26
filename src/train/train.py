def train_branch_predictor(model, train_loader, val_loader,
                          num_epochs=50, learning_rate=0.001,
                          device='cuda', patience=10):
    """
    Training loop with validation and early stopping

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: maximum number of epochs
        learning_rate: learning rate 
        device: 'cuda' 
        patience: early stopping patience (epochs without improvement)

    Returns:
        history: dict with training history
        best_model_path: path to saved best model
    """

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
        )

    #Initialize history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    best_model_path = f'{model.__class__.__name__}_best_state.pth'

    print(f"Training Phase")
    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {train_loader.batch_size}")

    for epoch in range(num_epochs):
        #Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (batch_histories, batch_targets) in enumerate(train_loader):
            batch_histories = batch_histories.to(device)
            batch_targets = batch_targets.to(device)

            #Forward pass
            predictions = model(batch_histories)
            loss = criterion(predictions, batch_targets)

            #Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Track metrics
            train_loss += loss.item()
            predicted = (predictions > 0.5).float()
            train_correct += (predicted == batch_targets).sum().item()
            train_total += batch_targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        #Validation phase
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

        #Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['learning_rate'].append(current_lr)

        #Learning rate scheduling
        scheduler.step(val_accuracy)

        #Save best model & early stopping if needed
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }, best_model_path)
            print(f"Saved model parameters. Val Accuracy: {val_accuracy:.4f}")
        else:
            epochs_without_improvement += 1

        #Print epoch summary
        print(f"\n  Epoch [{epoch+1}/{num_epochs}] Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}, Val Acc:   {val_accuracy:.4f}")
        print(f"    Best Val Acc: {best_val_accuracy:.4f}")
        print(f"    Epochs without improvement: {epochs_without_improvement}/{patience}")

        #Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early Stopping - No improvement for {patience} epochs")
            break

    print(f"Training Complete")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return history, best_model_path