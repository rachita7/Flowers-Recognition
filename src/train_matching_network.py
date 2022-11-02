import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import tqdm
from networks.matching import MatchingNetwork
from utils.plot import plot_curve

from utils.task_sampler import TaskSampler


def evaluate_model(model, criterion, data_loader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    running_loss = 0.0
    running_correct = 0
    total = 0
    
    for support_images, support_labels, query_images, query_labels, _ in tqdm.tqdm(data_loader):
        
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        scores = model(support_images, support_labels, query_images)

        loss = criterion(scores, query_labels)

        running_loss += loss.item()
        total += query_labels.shape[0]
        _, preds = torch.max(scores, 1)
        running_correct += torch.sum(preds == query_labels).item()


    print(f'Loss: {running_loss / len(data_loader)}, Accuracy: {running_correct / total}')
    
    return running_loss / len(data_loader), running_correct / total


def train(model, optimizer, criterion, train_loader, val_loader, num_epochs=100):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        start_time = time.time()
        for support_images, support_labels, query_images, query_labels, _ in tqdm.tqdm(train_loader):
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            scores = model(support_images, support_labels, query_images)
            
            loss = criterion(scores, query_labels)
            
            running_loss += loss.item()
            total += query_labels.shape[0]
            _, preds = torch.max(scores, 1)
            running_correct += torch.sum(preds == query_labels).item()
            
            loss.backward()
            optimizer.step()
         
        end_time = time.time()
        
        print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {running_correct / total}, Time: {(end_time - start_time):.4f}s')
        
        train_loss_history.append(running_loss / len(train_loader))
        train_acc_history.append(running_correct / total)
        
        val_loss, val_acc = evaluate_model(model, criterion, val_loader)
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


if __name__ == '__main__':
    
    image_size = 224
    
    data_directory = '../data'
    
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
    
    n_way = 10
    k_shot = 3
    n_query = 5
    n_tasks = 1000
    n_epochs = 100
    n_val_tasks = 100
    n_test_tasks = 1000
    
    training_set.get_labels = lambda: [
        instance for instance in training_set._labels
    ]
    
    validation_set.get_labels = lambda: [
        instance for instance in validation_set._labels
    ]
    
    test_set.get_labels = lambda: [
        instance for instance in test_set._labels
    ]
    
    train_sampler = TaskSampler(training_set, n_way, k_shot, n_query, n_tasks)
    validation_sampler = TaskSampler(validation_set, n_way, k_shot, n_query, n_val_tasks)
    test_sampler = TaskSampler(test_set, n_way, k_shot, n_query, n_test_tasks)
    
    train_loader = torch.utils.data.DataLoader(
        training_set,
        batch_sampler=train_sampler,
        collate_fn=train_sampler.collate_fn,
    )
    
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_sampler=validation_sampler,
        collate_fn=validation_sampler.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_sampler=test_sampler,
        collate_fn=test_sampler.collate_fn,
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = MatchingNetwork(image_size).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = train(model, optimizer, criterion, train_loader, validation_loader, num_epochs=n_epochs)
    
    train_loss_history, train_acc_histroy, val_loss_history, val_acc_history = history
    
    print('Train: ', end='')
    evaluate_model(model, criterion, train_loader)

    print('Validation: ', end='')
    evaluate_model(model, criterion, validation_loader)

    print('Test: ', end='')
    evaluate_model(model, criterion, test_loader)
    
    torch.save(model.state_dict(), f"../weights/matching_network.pt")
    
    plot_curve(train_loss_history, val_loss_history, title='Matching Network Loss', ylabel='Loss', legend_loc='upper right')
    plot_curve(train_acc_histroy, val_acc_history, title='Matching Network Accuracy', ylabel='Accuracy')
