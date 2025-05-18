import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm

# Import your custom modules
from data_loader import AudioDataset, load_data
from early_stopping import EarlyStopping

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
config = {
    "base_dir": "Z:/archive/archive/archive",  # Update this to your dataset path
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "patience": 10,
    "min_delta": 0.001,
    "image_size": (224, 224),
    "checkpoint_dir": "./checkpoints",
    "results_dir": "./results"
}

# Create directories if they don't exist
os.makedirs(config["checkpoint_dir"], exist_ok=True)
os.makedirs(config["results_dir"], exist_ok=True)

def train_model(model_name, model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=100):
    """
    Train a model with early stopping
    """
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config["patience"],
        min_delta=config["min_delta"],
        restore_best_weights=True
    )
    
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # No gradient calculation for validation
        with torch.no_grad():
            # Progress bar for validation
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        
        # Update learning rate scheduler if provided
        if scheduler:
            scheduler.step(epoch_val_loss)
        
        # Save checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                'train_acc': epoch_train_acc,
                'val_acc': epoch_val_acc
            }, f"{config['checkpoint_dir']}/{model_name}_checkpoint_epoch_{epoch+1}.pth")
        
        # Early stopping check
        early_stopping(epoch_val_acc, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best model weights
    early_stopping.restore_model(model)
    
    # Save final model
    torch.save(model.state_dict(), f"{config['checkpoint_dir']}/{model_name}_best.pth")
    
    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name)
    
    return model, train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, test_loader, class_names, model_name):
    """
    Evaluate model on test set and create confusion matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/{model_name}_confusion_matrix.png")
    
    # Create and plot normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Normalized Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/{model_name}_confusion_matrix_normalized.png")
    
    # Calculate accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"{model_name} Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, cm

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    """
    Plot training and validation curves
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{config['results_dir']}/{model_name}_training_curves.png")
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Loading data...")
    try:
        train_data, val_data, test_data = load_data(base_dir=config["base_dir"])
        
        # Create datasets
        train_dataset = AudioDataset(train_data, config["base_dir"], config["image_size"])
        val_dataset = AudioDataset(val_data, config["base_dir"], config["image_size"])
        test_dataset = AudioDataset(test_data, config["base_dir"], config["image_size"])
        
        # Get class names
        class_names = train_dataset.classes
        num_classes = len(class_names)
        print(f"Number of classes: {num_classes}")
        
        # Create data loaders with error handling
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True if torch.cuda.is_available() else False
            )
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            # Try with fewer workers
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
        
        # Define models to train
        models_to_train = {}
        
        # Add DenseNet121
        try:
            models_to_train["densenet121"] = models.densenet121(weights='IMAGENET1K_V1')
        except:
            # Fallback for older PyTorch versions
            models_to_train["densenet121"] = models.densenet121(pretrained=True)
        
        # Add ResNet50
        try:
            models_to_train["resnet50"] = models.resnet50(weights='IMAGENET1K_V1')
        except:
            # Fallback for older PyTorch versions
            models_to_train["resnet50"] = models.resnet50(pretrained=True)
        
        # Add EfficientNetV2-S
        try:
            models_to_train["efficientnet_v2_s"] = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
        except Exception as e:
            print(f"Error loading EfficientNetV2-S: {e}")
            print("Trying alternative approach...")
            try:
                # Alternative import for older PyTorch versions
                from torchvision.models import efficientnet_v2_s
                models_to_train["efficientnet_v2_s"] = efficientnet_v2_s(pretrained=True)
            except Exception as e:
                print(f"Could not load EfficientNetV2-S: {e}")
                print("Skipping EfficientNetV2-S")
        
        # Modify the final layer of pretrained models
        if "densenet121" in models_to_train:
            models_to_train["densenet121"].classifier = nn.Linear(
                models_to_train["densenet121"].classifier.in_features, 
                num_classes
            )
        
        if "resnet50" in models_to_train:
            models_to_train["resnet50"].fc = nn.Linear(
                models_to_train["resnet50"].fc.in_features, 
                num_classes
            )
        
        if "efficientnet_v2_s" in models_to_train:
            try:
                # For newer PyTorch versions
                models_to_train["efficientnet_v2_s"].classifier[1] = nn.Linear(
                    models_to_train["efficientnet_v2_s"].classifier[1].in_features, 
                    num_classes
                )
            except:
                # For older PyTorch versions or different structure
                try:
                    in_features = models_to_train["efficientnet_v2_s"].classifier.in_features
                    models_to_train["efficientnet_v2_s"].classifier = nn.Linear(in_features, num_classes)
                except:
                    print("Could not modify EfficientNetV2-S classifier layer")
                    if "efficientnet_v2_s" in models_to_train:
                        del models_to_train["efficientnet_v2_s"]
        
        # Train each model
        results = {}
        for model_name, model in models_to_train.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name} model")
            print(f"{'='*50}")
            
            # Move model to device
            model = model.to(device)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            )
            
            # Train model
            start_time = time.time()
            try:
                model, train_losses, val_losses, train_accs, val_accs = train_model(
                    model_name, model, train_loader, val_loader, criterion, optimizer, 
                    scheduler, config["num_epochs"]
                )
                training_time = time.time() - start_time
                
                # Evaluate model
                print(f"\nEvaluating {model_name} model on test set...")
                test_acc, confusion_mat = evaluate_model(model, test_loader, class_names, model_name)
                
                # Store results
                results[model_name] = {
                    "test_accuracy": test_acc,
                    "training_time": training_time,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accs": train_accs,
                    "val_accs": val_accs,
                    "confusion_matrix": confusion_mat
                }
                
                print(f"{model_name} training completed in {training_time:.2f} seconds")
                print(f"Test Accuracy: {test_acc:.2f}%")
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        # Compare model performances
        if results:
            print("\n" + "="*50)
            print("Model Performance Comparison")
            print("="*50)
            
            for model_name, result in results.items():
                print(f"{model_name}: Test Accuracy = {result['test_accuracy']:.2f}%, Training Time = {result['training_time']:.2f}s")
            
            # Find best model
            best_model = max(results.items(), key=lambda x: x[1]["test_accuracy"])
            print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['test_accuracy']:.2f}%")
            
            # Plot comparison of validation accuracies
            plt.figure(figsize=(10, 6))
            for model_name, result in results.items():
                plt.plot(result["val_accs"], label=model_name)
            
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy (%)")
            plt.title("Validation Accuracy Comparison")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{config['results_dir']}/model_comparison.png")
            plt.close()
        else:
            print("No models were successfully trained.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
