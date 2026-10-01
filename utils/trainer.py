import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from utils.metrics import calculate_metrics
from utils.optimizers import create_optimizer


def train_model(model, train_loader, val_loader, test_loader, num_epochs=50,
                learning_rate=0.001, device='cuda', patience=10, run_dir=None, 
                checkpoint_interval=10, optimizer_name='adam', optimizer_params=None,
                num_classes=None):
    """Train the model with optional validation and early stopping"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer with specified parameters
    if optimizer_params is None:
        optimizer_params = {}
    
    optimizer = create_optimizer(
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        **optimizer_params
    )
    
    scheduler = None
    if val_loader is not None:
        # Scheduler still monitors loss for smooth reduction, which is standard practice
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize training state
    best_val_acc = 0.0
    best_val_loss = float('inf')  # Track best loss for tie-breaking
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0
    start_epoch = 0
    
    training_log = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training on: {device}")
    print(f"Optimizer: {optimizer_name.upper()}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Using validation: {'Yes' if val_loader is not None else 'No'}")
    print(f"Saving best model based on: Validation Accuracy")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * train_correct / train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            with torch.no_grad():
                for inputs, targets in val_pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
                    
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100. * val_correct / val_total:.2f}%'
                    })
            
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Step scheduler based on loss (standard practice for optimization)
            scheduler.step(val_loss)
            
            # Save Best Model based on ACCURACY with LOSS as tie-breaker
            # Priority: Higher accuracy > Lower loss (when accuracy is same)
            save_model_flag = False
            improvement_msg = ""
            
            if val_acc > best_val_acc:
                # Better accuracy - always save
                save_model_flag = True
                improvement_msg = f"New best accuracy: {val_acc:.2f}% (was {best_val_acc:.2f}%)"
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_without_improvement = 0
            elif val_acc == best_val_acc and val_loss < best_val_loss:
                # Same accuracy but better loss - save as tie-breaker
                save_model_flag = True
                improvement_msg = f"Same accuracy ({val_acc:.2f}%) but better loss: {val_loss:.4f} (was {best_val_loss:.4f})"
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if save_model_flag:
                print(f"  {improvement_msg}")
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                if run_dir:
                    save_model(model, run_dir, 'best_model', verbose=False)
            
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }
            training_log.append(log_entry)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                  f"No improve: {epochs_without_improvement}/{patience}")
            
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
                break
        else:
            # If no validation set, just save the latest as best
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            if run_dir:
                save_model(model, run_dir, 'best_model', verbose=False)
            
            log_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc
            }
            training_log.append(log_entry)
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}%")
        
        if run_dir and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch + 1, best_val_acc, run_dir, scheduler, best_val_loss)
    
    training_time = time.time() - start_time
    
    if run_dir:
        # Save the very last state as final_model
        save_model(model, run_dir, 'final_model', verbose=True)
    
    # MODIFICATION 3: Explicitly load best model for testing
    # This ensures we test on the best epoch, not the last ran epoch
    print(f"\nReloading best model from epoch {best_epoch} for testing...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Evaluating on test set (Using Best Model)...")
    print(f"{'='*60}\n")
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_targets = []
    
    test_pbar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            test_pbar.set_postfix({
                'acc': f'{100. * test_correct / test_total:.2f}%'
            })
    
    # Use passed num_classes so rare classes with 0 test samples aren't silently dropped.
    # Fall back to inferring from targets only if num_classes was not provided.
    if num_classes is None:
        num_classes = len(set(all_targets))
    oa, aa, kappa, per_class_acc = calculate_metrics(all_preds, all_targets, num_classes)

    print(f"\nOverall Test Accuracy (OA): {oa:.2f}%")
    print(f"Average Test Accuracy (AA): {aa:.2f}%")
    print(f"Kappa: {kappa:.4f}")
    print(f"\n{'='*60}")
    print("Class-wise Test Accuracies:")
    print(f"{'='*60}")
    for i, acc in enumerate(per_class_acc):
        print(f"Class {i+1}: {acc:.2f}%")
    print(f"{'='*60}\n")
    
    # Save training log
    if run_dir:
        log_path = os.path.join(run_dir, 'training.log')
        with open(log_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Training Log\n")
            f.write("="*60 + "\n\n")
            for entry in training_log:
                f.write(str(entry) + '\n')
            f.write("\n" + "="*60 + "\n")
            f.write(f"Test Results (Best Model from Epoch {best_epoch})\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Test Accuracy (OA): {oa:.2f}%\n")
            f.write(f"Average Test Accuracy (AA): {aa:.2f}%\n")
            f.write(f"Kappa: {kappa:.4f}\n\n")
            f.write("Class-wise Test Accuracies:\n")
            f.write("-"*60 + "\n")
            for i, acc in enumerate(per_class_acc):
                f.write(f"Class {i+1}: {acc:.2f}%\n")
            f.write("="*60 + "\n")
        
        print(f"\n{'='*60}")
        print("Models & Logs Saved:")
        print(f"{'='*60}")
        print(f"best_model.pth (epoch {best_epoch}, Acc: {best_val_acc:.2f}%, Loss: {best_val_loss:.4f})")
        print(f"final_model.pth")
        
        checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            print(f"{checkpoints[0]}")
        print(f"training.log")
        print(f"{'='*60}\n")
    
    return model, all_preds, all_targets, best_epoch, training_time


def save_checkpoint(model, optimizer, epoch, best_val_acc, run_dir, scheduler=None, best_val_loss=float('inf')):
    """Save full checkpoint with optimizer state for resuming training"""
    # Remove old checkpoints
    old_checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    for old_ckpt in old_checkpoints:
        os.remove(os.path.join(run_dir, old_ckpt))
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: epoch_{epoch}.pth")


def save_model(model, run_dir, name='best_model', verbose=True):
    """Save model weights"""
    model_path = os.path.join(run_dir, f'{name}.pth')
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"Model saved: {name}.pth")


def print_model_summary(model, input_shape, device='cuda', depth=4):
    """Print model summary using torchinfo"""
    try:
        from torchinfo import summary
        
        print("\n" + "="*60)
        print("Model Summary")
        print("="*60)
        
        summary(
            model,
            input_size=input_shape,
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=depth,
            device=device
        )
        
        print("="*60 + "\n")
    except ImportError:
        print("Warning: torchinfo not installed. Install with: pip install torchinfo")
