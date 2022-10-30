import torch

class FlowerClassifier(torch.nn.Module):
    def __init__(self, num_classes=102, image_size=224) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.image_size = image_size
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(image_size * image_size * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Softmax()
        )
        
    def forward(self, x):
        return self.classifier(x)