import os
import argparse
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory')
    args = parser.parse_args()

    directory = args.directory

    if not os.path.isdir(directory):
        os.makedirs(directory)

    torchvision.datasets.Flowers102(root=f'{directory}', download=True)
    
    training_set = torchvision.datasets.Flowers102(root=f'{directory}', split='train')
    validation_set = torchvision.datasets.Flowers102(root=f'{directory}', split='val')
    test_set = torchvision.datasets.Flowers102(root=f'{directory}', split='test')
    
    print(f'Length of training data: {len(training_set)}')
    print(f'Length of validation data: {len(validation_set)}')
    print(f'Length of test data: {len(test_set)}')