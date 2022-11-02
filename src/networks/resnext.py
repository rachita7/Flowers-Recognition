import torch
import torchvision

class FlowerClassifier(torch.nn.Module):
    def __init__(self, num_classes=102, image_size=224, num_channels=3) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes),
        )
        
        self.model.fc = self.classifier
        
    def forward(self, x):
        return self.model(x)