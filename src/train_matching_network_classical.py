import argparse
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
    
    start_time = time.time()
    
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

    end_time = time.time()

    print(f'Loss: {running_loss / len(data_loader)}, Accuracy: {running_correct / total}, {(end_time - start_time):.4f}s')
    
    return running_loss / len(data_loader), running_correct / total


def train(backbone, fc_layer, model, optimizer, criterion, train_loader, val_loader, num_epochs=100):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        
        backbone.train()
        fc_layer.train()
        
        running_loss = 0.0
        running_correct = 0
        total = 0
        
        start_time = time.time()
        for images, labels in tqdm.tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            embedding = backbone(images)
            output = fc_layer(embedding)
            
            loss = criterion(output, labels)
            
            running_loss += loss.item()
            total += labels.shape[0]
            _, preds = torch.max(output, 1)
            running_correct += torch.sum(preds == labels).item()
            
            loss.backward()
            optimizer.step()
         
        end_time = time.time()
        
        print(f'Epoch: {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {running_correct / total}, Time: {(end_time - start_time):.4f}s')
        
        train_loss_history.append(running_loss / len(train_loader))
        train_acc_history.append(running_correct / total)
        
        backbone.eval()
        val_loss, val_acc = evaluate_model(model, criterion, val_loader)
        
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
    return train_loss_history, train_acc_history, val_loss_history, val_acc_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', default=224, type=int)
    parser.add_argument('--data-directory', default='../data')
    parser.add_argument('--model-directory', default='../weights')
    parser.add_argument('--use-fce', default=False, type=bool)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=3, type=int)
    parser.add_argument('--n_query', default=5, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--val-tasks', default=100, type=int)
    parser.add_argument('--test-tasks', default=1000, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    image_size = args.image_size
    data_directory = args.data_directory
    model_directory = args.model_directory
    use_fce = args.use_fce
    n_way = args.n_way
    k_shot = args.k_shot
    n_query = args.n_query
    n_val_tasks = args.val_tasks
    n_test_tasks = args.test_tasks
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    
    
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
    
    # n_way = 5
    # k_shot = 3
    # n_query = 5
    # n_tasks = 100
    # n_epochs = 100
    # n_val_tasks = 100
    # n_test_tasks = 1000
    
    training_set.get_labels = lambda: [
        instance for instance in training_set._labels
    ]
    
    validation_set.get_labels = lambda: [
        instance for instance in validation_set._labels
    ]
    
    test_set.get_labels = lambda: [
        instance for instance in test_set._labels
    ]
    
    validation_sampler = TaskSampler(validation_set, n_way, k_shot, n_query, n_val_tasks)
    test_sampler = TaskSampler(test_set, n_way, k_shot, n_query, n_test_tasks)
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True)
    
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
    
    backbone = torchvision.models.resnet18(pretrained=True)
    backbone.fc = torch.nn.Flatten()
    fully_connected_layer = torch.nn.Linear(in_features=512, out_features=102)
    
    model = MatchingNetwork(image_size=image_size, use_full_contextual_embedding=False, backbone=backbone).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(backbone.parameters()) + list(fully_connected_layer.parameters()), lr=0.01)
    
    history = train(backbone, fully_connected_layer, model, optimizer, criterion, train_loader, validation_loader, num_epochs=n_epochs)
    
    train_loss_history, train_acc_histroy, val_loss_history, val_acc_history = history
    
    # print('Train: ', end='')
    # evaluate_model(model, criterion, train_loader)

    print('Validation: ', end='')
    evaluate_model(model, criterion, validation_loader)

    print('Test: ', end='')
    evaluate_model(model, criterion, test_loader)
    
    torch.save(model.state_dict(), f"../weights/matching_network_classical.pt")
    
    plot_curve(train_loss_history, val_loss_history, title='Matching Network Loss', ylabel='Loss', legend_loc='upper right')
    plot_curve(train_acc_histroy, val_acc_history, title='Matching Network Accuracy', ylabel='Accuracy')
