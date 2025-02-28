import yaml
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from collections import Counter

import torch.nn as nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import torch 
import math

def compute_class_weights(train_loader, num_classes, device):
    class_counts = [0] * num_classes
    for batch_data, y in train_loader:
        # batch_data is [x1, x2]
        y = y.view(-1)  # Flatten to [batch_size * 6]
        counts = Counter(y.tolist())
        for cls in range(num_classes):
            class_counts[cls] += counts.get(cls, 0)
    total_counts = sum(class_counts)
    class_weights = [total_counts / (num_classes * count) if count > 0 else 0.0 for count in class_counts]
    return torch.tensor(class_weights, dtype=torch.float32).to(device)


def training(train_loader, 
             val_loader, 
             model, 
             optimizer, 
             scheduler, 
             criterion, 
             max_epochs, 
             num_classes,
             device,
             log_path):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 30
    no_improve_epochs = 0

    # Training loop
    max_grad_norm = 1.0  # Maximum norm for gradients

    for epoch in tqdm(range(max_epochs)):
        model.train()
        total_loss = 0.0
        for batch_data, y in train_loader:
            x1, x2 = batch_data  # Unpack x1 and x2
            
            # Convert inputs to float and move to device
            x1 = x1.to(device).float()
            x2 = x2.to(device).float()
            
            # Convert targets to long and move to device
            y = y.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(x1, x2)  # [batch_size, 6, num_classes]
            
            # Compute loss across all timesteps
            loss = criterion(outputs.reshape(-1, num_classes), y.reshape(-1))
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Average training loss
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation on validation data
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            correct_per_class = [0] * num_classes
            total_per_class = [0] * num_classes
            for batch_data, y in val_loader:
                x1, x2 = batch_data
                x1 = x1.to(device).float()
                x2 = x2.to(device).float()
                y = y.to(device).long()
                
                outputs = model(x1, x2)
                
                # Compute loss
                loss = criterion(outputs.reshape(-1, num_classes), y.reshape(-1))
                total_val_loss += loss.item()
                
                # Compute accuracy
                preds = outputs.argmax(dim=-1)  # [batch_size, 6]
                for t in range(6):
                    pred_t = preds[:, t]
                    target_t = y[:, t]
                    for cls in range(num_classes):
                        cls_mask = (target_t == cls)
                        cls_total = cls_mask.sum().item()
                        if cls_total > 0:
                            cls_correct = (pred_t[cls_mask] == target_t[cls_mask]).sum().item()
                            correct_per_class[cls] += cls_correct
                            total_per_class[cls] += cls_total
            # Compute class-balanced accuracy
            class_accuracies = []
            for cls in range(num_classes):
                if total_per_class[cls] > 0:
                    acc = 100.0 * correct_per_class[cls] / total_per_class[cls]
                else:
                    acc = 0.0
                class_accuracies.append(acc)
            class_balanced_acc = sum(class_accuracies) / num_classes
            
            avg_val_loss = total_val_loss / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Class-balanced Accuracy: {class_balanced_acc:.2f}%')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = class_balanced_acc
            no_improve_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), f'{log_path}/best_model.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return best_val_loss, best_val_acc, model

def accuracy_per_timestep(model, test_loader, num_classes, device):

    model.eval()
    with torch.no_grad():
        accuracy_per_timestep_test = []
        for t in range(6):
            correct_per_class = [0] * num_classes
            total_per_class = [0] * num_classes
            for batch_data, y in test_loader:
                x1, x2 = batch_data
                x1 = x1.to(device).float()
                x2 = x2.to(device).float()
                y = y.to(device).long()
                
                outputs = model(x1, x2)  # Inference mode
                preds = outputs.argmax(dim=-1)  # [batch_size, 6]
                pred_t = preds[:, t]
                target_t = y[:, t]
                
                for cls in range(num_classes):
                    cls_mask = (target_t == cls)
                    cls_total = cls_mask.sum().item()
                    if cls_total > 0:
                        cls_correct = (pred_t[cls_mask] == target_t[cls_mask]).sum().item()
                        correct_per_class[cls] += cls_correct
                        total_per_class[cls] += cls_total
            # Compute class-balanced accuracy
            class_accuracies = []
            for cls in range(num_classes):
                if total_per_class[cls] > 0:
                    acc = 100.0 * correct_per_class[cls] / total_per_class[cls]
                else:
                    acc = 0.0
                class_accuracies.append(acc)
            class_balanced_acc = sum(class_accuracies) / num_classes
            accuracy_per_timestep_test.append(class_balanced_acc)
            
    return accuracy_per_timestep_test