import torch
import torchvision

class FlowerClassifier(torch.nn.Module):
    def __init__(self, num_classes=102, image_size=224, num_channels=3) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_channels = num_channels
        
        self.model = torchvision.models.resnet18(pretrained=True)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.model.fc.in_features, num_classes)
        )
        
        self.model.fc = self.classifier
        
    def forward(self, x):
        return self.model(x)