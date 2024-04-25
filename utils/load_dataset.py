import torch
import torchvision
import torchvision.transforms as transforms

# Function to load CIFAR-10 dataset
def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # EfficientNetV2-S expects input size 224x224
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet stats
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    return trainloader, testloader