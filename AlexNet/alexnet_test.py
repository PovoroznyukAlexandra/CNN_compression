import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from alexnet_train import AlexNet

def test_accuracy(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 100, "\n",)

def model_compression_ratio(original_model, compressed_model):
    original_total = sum(p.numel() for p in original_model.parameters())
    compressed_total = sum(p.numel() for p in compressed_model.parameters())
    print(f"Original_total = {original_total}")
    print(f"Compressed_total = {compressed_total}")
    print(f"Percent of compresson = {(1 - compressed_total / original_total) * 100:.2f}%")
    print(f"Compressed into {(original_total / compressed_total):.2f} times")
    print("=" * 100, "\n")

if __name__ == '__main__':
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load("weights/alexnet_cifar10_100epoch.pth"))
    test_accuracy(model)
  