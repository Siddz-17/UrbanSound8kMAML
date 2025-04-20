from data_loader import load_data, AudioDataset
from model import MAMLModel
from trainer import MAMLTrainer
import torch
import os
from early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
def main():
    # Configuration
    config = {
        "base_dir": "Z:/archive/archive/archive",  # Updated base directory
        "csv_path": None,  # Set to None to use directory scanning
        "image_size": (224, 224),
        "num_classes_per_set": 5,
        "num_samples_per_class": 1,
        "num_target_samples": 5,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Load and prepare data
    train_data, val_data, test_data = load_data(
        config['csv_path'],
        config['base_dir']
    )
    
    train_dataset = AudioDataset(
        train_data,
        config['base_dir'],
        config['image_size']
    )
    
    val_dataset = AudioDataset(
        val_data,
        config['base_dir'],
        config['image_size']
    )
    
    # Initialize model
    num_classes = len(set(s['class'] for s in train_data))
    model = MAMLModel(num_classes)
    
    # Initialize trainer
    trainer = MAMLTrainer(model, config['device'])
    
    # Training loop
    for epoch in range(config['num_epochs']):
        train_loss, train_acc = trainer.train_epoch(train_dataset)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        if (epoch+1) % 5 == 0:
            val_loss, val_acc = trainer.validate(val_dataset)
            print(f"Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save model checkpoint
            checkpoint_dir = os.path.join(config['base_dir'], 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'model_epoch{epoch+1}.pth'))

if __name__ == "__main__":
    main()
