import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from collections import OrderedDict


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)),
            ('relu1', nn.ReLU(inplace=True)),
            ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            
            ('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU(inplace=True)),
            ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            
            ('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            
            ('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            
            ('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('maxpool3', nn.MaxPool2d(kernel_size=3, stride=2)),
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(0.5)),
            ('linear1', nn.Linear(256 * 6 * 6, 4096)),
            ('relu6', nn.ReLU(inplace=True)),
            
            ('dropout2', nn.Dropout(0.5)),
            ('linear2', nn.Linear(4096, 4096)),
            ('relu7', nn.ReLU(inplace=True)),
            
            ('linear3', nn.Linear(4096, num_classes))
        ]))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_train = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(227, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    
    model = AlexNet(num_classes=10)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_accuracy = 0
    
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")
        
    torch.save(model.state_dict(), "weights/alexnet_cifar10_100epochs.pth")

if __name__ == '__main__':
    train()