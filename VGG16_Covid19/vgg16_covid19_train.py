import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import multiprocessing
from collections import OrderedDict
from tqdm import tqdm


class VGG16(nn.Module):
    def __init__(self, num_classes=3):
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
            ('linear1', nn.Linear(512 * 7 * 7, 4096)),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.15245409309864044, 0.15343371033668518, 0.1577882021665573],
                            std=[0.15406405925750732, 0.15451762080192566, 0.15620222687721252])
    ])

    trainset = datasets.ImageFolder(root='./split_dataset/dataset_train', transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    
    model = VGG16(num_classes=3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

    num_epochs = 5
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

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "weights/vgg16_covid19.pth")

    data_iter = iter(trainloader)
    images, labels = next(data_iter)
    with open('results/train_parameters.txt', 'w') as f:
        f.write(f"Количество эпох: {num_epochs}\n")
        f.write(f"Размер одного изображения: {images[0].shape}\n")
        f.write(f"Размер батча: {images.shape[0]}\n")
        f.write(f"Тип данных: {images.dtype}\n")
        f.write(f"Размер датасета: {len(trainset)} изображений\n")
        f.write(f"Количество батчей: {len(trainloader)}\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()