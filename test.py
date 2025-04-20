import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
from model import MAMLModel
from data_loader import load_data, AudioDataset

# Configuration
config = {
    "base_dir": "Z:/archive/archive/archive",  # Updated base directory
    "csv_path": None,  # Set to None to use directory scanning 
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "image_size": (224, 224),
    "num_classes_per_set": 5,
    "num_samples_per_class": 1,
    "num_target_samples": 5,
    "batch_size": 32,
    "num_epochs": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# MAML Trainer
class MAML:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, task, num_steps=5):
        fast_weights = dict(self.model.named_parameters())
        for _ in range(num_steps):
            outputs = self.model(task['support_x'], params=fast_weights)
            loss = self.loss_fn(outputs, task['support_y'])
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = {
                name: param - 0.01 * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        return fast_weights

    def evaluate(self, task, fast_weights):
        with torch.no_grad():
            outputs = self.model(task['query_x'], params=fast_weights)
            loss = self.loss_fn(outputs, task['query_y'])
            preds = outputs.argmax(dim=1)
            accuracy = (preds == task['query_y']).float().mean()
        return loss.item(), accuracy.item()

    def train_epoch(self, dataset, num_tasks=100):
        self.model.train()
        total_loss = 0
        total_acc = 0
        successful_tasks = 0
        
        for _ in range(num_tasks):
            try:
                task = dataset.get_task_batch()
                task = {k: v.to(self.device) for k, v in task.items()}
                
                fast_weights = self.adapt(task)
                
                self.optimizer.zero_grad()
                outputs = self.model(task['query_x'], params=fast_weights)
                loss = self.loss_fn(outputs, task['query_y'])
                loss.backward()
                self.optimizer.step()
                
                preds = outputs.argmax(dim=1)
                acc = (preds == task['query_y']).float().mean().item()
                
                total_loss += loss.item()
                total_acc += acc
                successful_tasks += 1
            except Exception as e:
                print(f"Error during training: {e}")
                continue
                
        if successful_tasks == 0:
            raise ValueError("No tasks were successfully processed")
            
        return total_loss / successful_tasks, total_acc / successful_tasks

# Main Training Loop
def main():
    # Load and split data
    train_data, val_data, test_data = load_data(
        config['csv_path'], 
        config['base_dir'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio']
    )

    # Create datasets
    train_dataset = AudioDataset(train_data, config['base_dir'], config['image_size'])
    val_dataset = AudioDataset(val_data, config['base_dir'], config['image_size'])
    test_dataset = AudioDataset(test_data, config['base_dir'], config['image_size'])

    # Get unique classes
    num_classes = len(set(s['class'] for s in train_data))
    print(f"Found {num_classes} unique classes")

    # Initialize model and trainer
    model = MAMLModel(num_classes)
    maml = MAML(model, config['device'])

    # Training loop
    for epoch in range(config['num_epochs']):
        try:
            train_loss, train_acc = maml.train_epoch(train_dataset)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            
            # Validation
            if (epoch+1) % 5 == 0:
                val_loss, val_acc = 0, 0
                try:
                    val_loss, val_acc = maml.train_epoch(val_dataset, num_tasks=50)  # Fewer tasks for validation
                except Exception as e:
                    print(f"Validation error: {e}")
                print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # Save checkpoint
                checkpoint_dir = os.path.join(config['base_dir'], 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth'))
                
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            continue

if __name__ == "__main__":
    main()
