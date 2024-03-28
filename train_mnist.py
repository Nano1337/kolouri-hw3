import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from tqdm import tqdm

from vit import ViT
from cnn import CNN

# Define a transformation to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # MNIST images are grayscale
])

if __name__ == "__main__": 

    # settings
    batch_size = 128
    num_epochs = 15

    # Load the data
    train_val_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    num_train = len(train_val_dataset)
    num_val = int(num_train * 0.1)  # 10% for validation
    num_train -= num_val

    train_dataset, val_dataset = random_split(train_val_dataset, [num_train, num_val])

    # Create data loaders for each set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    models = {
        'vit': ViT(
            img_size=28,
            patch_size=7,
            in_channels=1,
            num_classes=10,
            embed_size=64,
            depth=6,
            heads=8,
            forward_expansion=4,
            dropout=0.1
        ),
        'cnn': CNN(),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()

    losses = {}
    accuracies = {}

    for model_type, model in models.items():
        print(f'Training {model_type} model')
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
        model.train()

        losses[model_type] = []
        accuracies[model_type] = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = train_loader
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                losses[model_type].append(epoch_loss)
                accuracies[model_type].append(epoch_acc.cpu())

        print('Finished Training')

        # Test the model
        model.eval() 
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in tqdm(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Plotting the losses
    plt.figure(figsize=(10, 5))
    for model_type, loss in losses.items():
        plt.plot(loss, label=model_type)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss.png')

    # Plotting the accuracies
    plt.figure(figsize=(10, 5))
    for model_type, accuracy in accuracies.items():
        plt.plot(accuracy, label=model_type)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('accuracy.png')
