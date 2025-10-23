import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict
import multiprocessing

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('batcnorm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('batcnorm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]))
        self.block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('batcnorm1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('batcnorm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]))
        self.block3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('batcnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('batcnorm2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('batcnorm3', nn.BatchNorm2d(256)),
            ('relu3', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]))
        self.block4 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ('batcnorm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('batcnorm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('batcnorm3', nn.BatchNorm2d(512)),
            ('relu3', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]))
        self.block5 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('batcnorm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('batcnorm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('batcnorm3', nn.BatchNorm2d(512)),
            ('relu3', nn.ReLU()),
            ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]))

        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout()),
            ('linear1', nn.Linear(512 * 1 * 1, 4096)),
            ('relu1', nn.ReLU()),
            ('dropout2', nn.Dropout()),
            ('linear2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU()),
            ('linear3', nn.Linear(4096, num_classes))]))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    model = VGG16(num_classes=10)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')), labels.to(
                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    torch.save(model.state_dict(), 'weights/vgg16.pth')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()