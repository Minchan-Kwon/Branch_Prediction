import sys
import argparse
import numpy as np
from pathlib import Path
from src.utils.BranchHistory import BranchHistory
from src.model.baseline import BaselineModel
from src.utils import data_utils
from src.model import cnn_models as cnn
from src.model import lstm_models as lstm
from src.train.train import train_branch_predictor
from src.train.qtrain import train_qat_model
from src.utils import utils
import torch
import traceback

def predict_baseline(args):
    # Create Global History Dataframe and Print Stats
    glob_history = BranchHistory.from_csv(args.csv)
    glob_history.print_stats()
    
    # Create Baseline Model and Make Prediction
    baseline_model = BaselineModel()
    baseline_model.predict(glob_history) # Prediction Saved to ../data/baseline_prediction.csv
    
def extract_branch_history(args):
    '''
    # Convert PC from string to int
    if args.pc.startswith('0x'): 
        args.pc = int(args.pc, 16)
    else: 
        args.pc = int(args.pc) '''
    
    # Global History Dataframe
    glob_history = BranchHistory.from_csv(args.csv)
    
    # Extract trainable history with respect to given branch PC
    print("Extracting Branch History for PC 0x{args.pc:x}")
    history_indices, targets, one_hot_dim = glob_history.extract_branch_history(
        branch_pc=args.pc,
        history_length=args.length,
        cutoff=args.cutoff)
    
    # Save extracted history, label and metadata
    output_dir = Path(f"./run/data/0x{args.pc:x}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving data to {output_dir}")
    np.save(output_dir/'histories.npy', history_indices)
    np.save(output_dir/'targets.npy', targets)
    vocab_size = 2 ** (args.cutoff + 1)

    with open(output_dir/'metadata.txt', 'w') as f:
        f.write(f"branch_pc: 0x{args.pc:x}\n")
        f.write(f"history_length: {args.length}\n")
        f.write(f"cutoff: {args.cutoff}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"num_samples: {len(history_indices)}\n")
        f.write(f"batches: {args.batch}\n")
    
def train(args):
    # Load history and label
    print(f"Loading data from {args.data_dir}")
    history_indices = np.load(f"{args.data_dir}/histories.npy")
    targets = np.load(f"{args.data_dir}/targets.npy")
    
    # Read metadata
    print(f"Loading metadata")
    metadata = {}
    with open(f'{args.data_dir}/metadata.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key == 'branch_pc':
                metadata[key] = int(value, 16)
            if key in ['vocab_size', 'num_samples', 'history_length', 'cutoff', 'batches']:
                metadata[key] = int(value)
    
    vocab_size = metadata['vocab_size']
    history_length = metadata['history_length']
    batch_size = metadata['batches']
    
    # Train, val, test split (Temporal)
    print("Splitting Data")
    train_hist, train_tgt, val_hist, val_tgt, test_hist, test_tgt = data_utils.split_data_temporal(
        history_indices, targets,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"  Train: {len(train_tgt):,} samples")
    print(f"  Val:   {len(val_tgt):,} samples")
    print(f"  Test:  {len(test_tgt):,} samples")
    
    # Create train, val, test loaders 
    print("Creating Dataloaders")
    train_loader, val_loader, test_loader = data_utils.create_dataloaders(
        train_data=(train_hist, train_tgt),
        val_data=(val_hist, val_tgt),
        test_data=(test_hist, test_tgt),
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Initialize Model
    print("\nInitializing Model")
    if args.model == "ultralight":
        model = cnn.CNN_Ultralight(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "light":
        model = cnn.CNN_Light(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "medium":
        model = cnn.CNN_Medium(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "heavy":
        model = cnn.CNN_Heavy(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "humongous":
        model = cnn.CNN_Humongous(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "lstm":
        model = lstm.LSTM_Medium(vocab_size=vocab_size).to(args.device)
    
    # Training
    train_history, best_model_path = train_branch_predictor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epoch,
        learning_rate=args.lr,
        device=args.device,
        patience=args.patience
    )
    print(f"Best Model Saved to {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    print("Plotting Training History")
    utils.plot_training_history(train_history)
    
    # Evaluate Model
    acc, prec, rec, f1 = utils.evaluate_model(model, test_loader, args.device)
    size = utils.get_model_size_mb(model)
    params = utils.count_parameters(model)
    print(f"Model Size (MB): {size}")
    print(f"Number of Parameters: {params}")

    
def qtrain(args):
    # Load history and label
    print(f"Loading data from {args.data_dir}")
    history_indices = np.load(f"{args.data_dir}/histories.npy")
    targets = np.load(f"{args.data_dir}/targets.npy")
    
    # Read metadata
    print(f"Loading metadata")
    metadata = {}
    with open(f'{args.data_dir}/metadata.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if key == 'branch_pc':
                metadata[key] = int(value, 16)
            if key in ['vocab_size', 'num_samples', 'history_length', 'cutoff', 'batches']:
                metadata[key] = int(value)
    
    vocab_size = metadata['vocab_size']
    history_length = metadata['history_length']
    batch_size = metadata['batches']
    
    # Train, val, test split (Temporal)
    print("Splitting Data")
    train_hist, train_tgt, val_hist, val_tgt, test_hist, test_tgt = data_utils.split_data_temporal(
        history_indices, targets,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"  Train: {len(train_tgt):,} samples")
    print(f"  Val:   {len(val_tgt):,} samples")
    print(f"  Test:  {len(test_tgt):,} samples")
    
    # Create train, val, test loaders 
    print("Creating Dataloaders")
    train_loader, val_loader, test_loader = data_utils.create_dataloaders(
        train_data=(train_hist, train_tgt),
        val_data=(val_hist, val_tgt),
        test_data=(test_hist, test_tgt),
        vocab_size=vocab_size,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Initialize Model
    print("\nInitializing Model")
    if args.model == "ultralight":
        model = cnn.CNN_Ultralight(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "light":
        model = cnn.CNN_Light(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "medium":
        model = cnn.CNN_Medium(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "heavy":
        model = cnn.CNN_Heavy(history_length=history_length, vocab_size = vocab_size).to(args.device)
    elif args.model == "humongous":
        model = cnn.CNN_Humongous(history_length=history_length, vocab_size = vocab_size).to(args.device)
    
    # Training
    train_history, best_model_path, _ = train_qat_model (
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epoch,
        learning_rate=args.lr,
        device=args.device,
        patience=args.patience
    )
    print(f"Best Model Saved to {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    print("Plotting Training History")
    utils.plot_training_history(train_history)
    
    # Evaluate Model
    acc, prec, rec, f1 = utils.evaluate_model(model, test_loader, args.device)
    size = utils.get_model_size_mb(model)
    params = utils.count_parameters(model)
    print(f"Model Size (MB): {size}")
    print(f"Number of Parameters: {params}")


def main(): 
    parser = argparse.ArgumentParser(description='DL Branch Prediction Pipeline', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline Stage')
    
    # predict_baseline
    baseline_parser = subparsers.add_parser('predict_baseline', help='Use Baseline Model to Make Prediction')
    baseline_parser.add_argument('--csv', required=True, help='Path to Global Branch History CSV')
    baseline_parser.set_defaults(func=predict_baseline)
    
    # extract_branch_history
    extract_parser = subparsers.add_parser('extract_branch_history', help='Extract Branch History and Create Dataloader')
    extract_parser.add_argument('--csv', required=True, help='Path to Global Branch History CSV')
    extract_parser.add_argument('--pc', required=True, type=lambda x: int(x, 16), help='Branch PC in Hexadecimal (e.g., 0x123456)') # PC is stored as decimal integer
    extract_parser.add_argument('--length', type=int, default=200, help='History Length')
    extract_parser.add_argument('--cutoff', type=int, default=8, help='PC Cutoff Bits')
    extract_parser.add_argument('--max_history', type=int, default=50000, help='History Length')
    extract_parser.add_argument('--batch', type=int, default=64, help='Batch Size of Dataset')
    extract_parser.set_defaults(func=extract_branch_history)
    
    # train
    train_parser = subparsers.add_parser('train', help='Train Model')
    train_parser.add_argument('--data_dir', required=True, help='Directory That Contains Saved Data')
    train_parser.add_argument('--model', choices=['ultralight', 'light', 'medium', 'heavy', 'humongous', 'lstm'], required=True)
    train_parser.add_argument('--epoch', type=int, default=50, help='Number of Epochs')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    train_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps', 'xpu', 'xla', 'meta'])
    train_parser.add_argument('--patience', type=int, default=10, help='Number of Epochs to Stop Training After No Decrease in Loss')
    train_parser.set_defaults(func=train)
    
    # qtrain
    qtrain_parser = subparsers.add_parser('qtrain', help='Quantization Aware Training')
    qtrain_parser.add_argument('--data_dir', required=True, help='Directory That Contains Saved Data')
    qtrain_parser.add_argument('--model', choices=['ultralight', 'light', 'medium', 'heavy', 'humongous'], required=True)
    qtrain_parser.add_argument('--epoch', type=int, default=50, help='Number of Epochs')
    qtrain_parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
    qtrain_parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps', 'xpu', 'xla', 'meta'])
    qtrain_parser.add_argument('--patience', type=int, default=10, help='Number of Epochs to Stop Training After No Decrease in Loss')
    qtrain_parser.set_defaults(func=qtrain)    
    
    args=parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
        
    # Exception Handler
    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
if __name__ == '__main__':
    main()