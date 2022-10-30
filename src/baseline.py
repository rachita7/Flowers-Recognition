import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


from networks.baseline import FlowerClassifier

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    args = parser.parse_args()

    data_directory = args.directory
    batch_size = 1020
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    
    training_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='train', transform=transform)
    validation_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='val', transform=transform)
    test_set = torchvision.datasets.Flowers102(root=f'{data_directory}', split='test', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                          shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_set),
                                          shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                            shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    classifier = FlowerClassifier(102, 224)
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss
        
        print(f'epoch [{epoch + 1}] loss: {running_loss / len(train_loader):.3f}')

    print('Finished Training')
