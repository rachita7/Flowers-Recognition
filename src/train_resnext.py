import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import tqdm

from networks.resnext import FlowerClassifier
from utils.plot import plot_curve


def validate(model, criterion, data_loader):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    
    start_time = time.time()
    num_samples = 0
    for inputs, labels in tqdm.tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_correct += torch.sum(preds == (labels)).item()
        
        num_samples += inputs.size(0)

    end_time = time.time()

    val_loss = running_loss / num_samples
    val_acc = running_correct / num_samples

    print(f'Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Time taken: {(end_time - start_time):.4f}s')
    
    return val_loss, val_acc


def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=40):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        
        model.train()
        
        running_loss = 0.0
        running_correct = 0
        
        start_time = time.time()
        num_samples = 0
        
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == (labels)).item()
            
            num_samples += inputs.size(0)
            
        end_time = time.time()
        
        epoch_loss = running_loss / num_samples
        epoch_acc = running_correct / num_samples
        
        print(f'Epoch [{epoch + 1}] Train loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}, Time taken: {(end_time - start_time):.4f}s')
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        print('Validation: ', end='')
        
        val_loss, val_acc = validate(model, criterion, val_loader)
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-directory')
    parser.add_argument('--model-directory')
    
    args = parser.parse_args()

    data_directory = args.data_directory
    model_directory = args.model_directory
    
    batch_size = 32
    image_size = 224
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ]
    )
    
    training_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='train', transform=transform)
    validation_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='val', transform=transform)
    test_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='test', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                          shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False)

    print(f'Length of training data: {len(training_set)}')
    print(f'Length of validation data: {len(validation_set)}')
    print(f'Length of test data: {len(test_set)}')
    
    criterion = torch.nn.CrossEntropyLoss()
    model = FlowerClassifier(num_classes=102, image_size=image_size)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    history = train(model, criterion, optimizer, train_loader, validation_loader)
    
    train_loss_history, train_acc_histroy, val_loss_history, val_acc_history = history
    
    print('Train: ', end='')
    validate(model, criterion, train_loader)

    print('Validation: ', end='')
    validate(model, criterion, validation_loader)

    print('Test: ', end='')
    validate(model, criterion, test_loader)
    
    torch.save(model.state_dict(), f"{model_directory}/resnext.pt")
    
    plot_curve(train_loss_history, val_loss_history, title='Resnext Loss', ylabel='Loss', legend_loc='upper right')
    plot_curve(train_acc_histroy, val_acc_history, title='Resnext Accuracy', ylabel='Accuracy')
    
