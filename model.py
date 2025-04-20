import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class MAMLModel(nn.Module):
    def __init__(self, num_classes, num_filters=64):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(3, num_filters),
            nn.MaxPool2d(2),
            ConvBlock(num_filters, num_filters),
            nn.MaxPool2d(2),
            ConvBlock(num_filters, num_filters),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(num_filters, num_classes)
        
    def forward(self, x, params=None):
        if params is None:
            params = dict(self.named_parameters())
            
        x = self._forward_features(x, params)
        x = x.mean([2, 3])  # Global average pooling
        x = F.linear(x, 
                    weight=params.get('classifier.weight', self.classifier.weight),
                    bias=params.get('classifier.bias', self.classifier.bias))
        return x
    
    def _forward_features(self, x, params):
        for i, block in enumerate(self.blocks):
            if isinstance(block, ConvBlock):
                weight = params.get(f'blocks.{i}.conv.weight', block.conv.weight)
                bias = params.get(f'blocks.{i}.conv.bias', block.conv.bias)
                x = F.conv2d(x, weight, bias, stride=1, padding=1)
                x = F.batch_norm(x, 
                                block.bn.running_mean, 
                                block.bn.running_var,
                                params.get(f'blocks.{i}.bn.weight', block.bn.weight),
                                params.get(f'blocks.{i}.bn.bias', block.bn.bias),
                                training=self.training)
                x = F.relu(x)
            else:
                x = F.max_pool2d(x, 2)
        return x