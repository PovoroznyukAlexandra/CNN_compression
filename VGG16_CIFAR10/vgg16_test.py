import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing

from vgg16_train import VGG16


def test_accuracy(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
def model_compression_ratio(original_model, compressed_model):
    original_total = sum(p.numel() for p in original_model.parameters())
    compressed_total = sum(p.numel() for p in compressed_model.parameters())
    print(f"Original_total = {original_total}")
    print(f"Compressed_total = {compressed_total}")
    print(f"Percent of compresson = {(1 - compressed_total / original_total) * 100:.2f}%")
    print(f"Compressed into {(original_total / compressed_total):.2f} times")
    print("=" * 100, "\n")

if __name__ == '__main__':
    model = VGG16(num_classes=10)
    model.load_state_dict(torch.load('weights/vgg16.pth'))
    test_accuracy(model)