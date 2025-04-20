import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
from collections import defaultdict
import torchvision.transforms as transforms

class AudioDataset(Dataset):
    def __init__(self, samples, base_dir, image_size=(224, 224)):
        self.samples = samples
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self._organize_samples()

    def _organize_samples(self):
        self.class_to_samples = defaultdict(list)
        for sample in self.samples:
            self.class_to_samples[sample['class']].append(sample)
        self.classes = list(self.class_to_samples.keys())
        
    def get_task_batch(self, num_classes_per_set=5, num_samples_per_class=1, num_target_samples=5):
        import random
        selected_classes = random.sample(self.classes, num_classes_per_set)
        support_set = []
        query_set = []
        
        for class_id in selected_classes:
            class_samples = self.class_to_samples[class_id]
            if len(class_samples) >= (num_samples_per_class + num_target_samples):
                selected = random.sample(class_samples, num_samples_per_class + num_target_samples)
                support_set.extend(selected[:num_samples_per_class])
                query_set.extend(selected[num_samples_per_class:])
        
        support_x, support_y = self._load_samples(support_set)
        query_x, query_y = self._load_samples(query_set)
        
        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y
        }
    
    def _load_samples(self, samples):
        import torch
        images = []
        labels = []
        label_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        for sample in samples:
            img_path = os.path.join(self.base_dir, 'images', 'train', sample['class'], sample['file_name'])
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
                labels.append(label_to_idx[sample['class']])
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
                
        if not images:
            raise ValueError("No images were successfully loaded")
            
        return torch.stack(images), torch.tensor(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.base_dir, 'images', 'train', sample['class'], sample['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            # Return a placeholder image or skip
            raise ValueError(f"Could not load image at {img_path}")
            
        if self.transform:
            image = self.transform(image)
            
        # Convert class name to index
        label_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        class_idx = label_to_idx[sample['class']]
        
        return image, class_idx

def load_data(csv_path=None, base_dir="Z:/archive/archive/archive", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    If CSV path is provided, load from there, otherwise scan the directory structure
    """
    if csv_path and os.path.exists(csv_path):
        # Load from CSV if available
        df = pd.read_csv(csv_path)
        samples = df[['file_name', 'class']].to_dict('records')
    else:
        # Scan directory structure
        samples = []
        images_dir = os.path.join(base_dir, 'images', 'train')
        
        for class_name in os.listdir(images_dir):
            class_dir = os.path.join(images_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        samples.append({
                            'file_name': file_name,
                            'class': class_name
                        })
    
    # First split into train+val and test
    train_val, test = train_test_split(
        samples,
        test_size=test_ratio,
        stratify=[s['class'] for s in samples]
    )
    
    # Then split train_val into train and val
    train, val = train_test_split(
        train_val,
        test_size=val_ratio/(train_ratio+val_ratio),
        stratify=[s['class'] for s in train_val]
    )
    
    return train, val, test
