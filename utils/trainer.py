import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


def train_model(model, train_loader, val_loader, test_loader, num_epochs=50,
                learning_rate=0.001, device='cuda', patience=10, run_dir=None, 
                checkpoint_interval=10):
    """Train the model with optional validation and early stopping"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if val_loader is not None:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0
    best_epoch = 0
    
    training_log = []
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Training on: {device}")
    print(f"Using validation: {'Yes' if val_loader is not None else 'No'}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
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
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                if run_dir:
                    save_model(model, run_dir, 'best_model', verbose=False)
            else:
                epochs_without_improvement += 1
            
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
                print("Restoring best model...")
                model.load_state_dict(best_model_state)
                break
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_model_state = model.state_dict().copy()
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
            save_checkpoint(model, run_dir, epoch + 1)
    
    training_time = time.time() - start_time
    
    if run_dir:
        save_model(model, run_dir, 'final_model', verbose=True)
    
    # Test evaluation
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
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
    
    test_acc = 100. * test_correct / test_total
    
    # Calculate class-wise accuracies
    num_classes = len(set(all_targets))
    class_wise_acc = []
    for class_id in range(num_classes):
        mask = np.array(all_targets) == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(np.array(all_targets)[mask], np.array(all_preds)[mask]) * 100
            class_wise_acc.append((class_id, class_acc, mask.sum()))
        else:
            class_wise_acc.append((class_id, 0.0, 0))
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\n{'='*60}")
    print("Class-wise Test Accuracies:")
    print(f"{'='*60}")
    for class_id, acc, count in class_wise_acc:
        print(f"Class {class_id+1}: {acc:.2f}% ({count} samples)")
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
            f.write("Test Results\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n\n")
            f.write("Class-wise Test Accuracies:\n")
            f.write("-"*60 + "\n")
            for class_id, acc, count in class_wise_acc:
                f.write(f"Class {class_id+1}: {acc:.2f}%\n")
            f.write("="*60 + "\n")
        
        print(f"\n{'='*60}")
        print("Models & Logs Saved:")
        print(f"{'='*60}")
        print(f"✓ best_model.pth (epoch {best_epoch})")
        print(f"✓ final_model.pth")
        
        checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_')]
        if checkpoints:
            print(f"✓ {checkpoints[0]}")
        print(f"✓ training.log")
        print(f"{'='*60}\n")
    
    return model, all_preds, all_targets, best_epoch, training_time


def save_checkpoint(model, run_dir, epoch):
    """Save model checkpoint"""
    old_checkpoints = [f for f in os.listdir(run_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    for old_ckpt in old_checkpoints:
        os.remove(os.path.join(run_dir, old_ckpt))
    
    checkpoint_path = os.path.join(run_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Checkpoint saved: epoch_{epoch}.pth")


def save_model(model, run_dir, name='best_model', verbose=True):
    """Save model weights"""
    model_path = os.path.join(run_dir, f'{name}.pth')
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"✓ Model saved: {name}.pth")
