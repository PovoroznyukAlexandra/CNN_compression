import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from vgg16_covid19_train import VGG16

def test_accuracy(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.15245409309864044, 0.15343371033668518, 0.1577882021665573],
                            std=[0.15406405925750732, 0.15451762080192566, 0.15620222687721252])
    ])

    testset = datasets.ImageFolder(root='./split_dataset/dataset_test', transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

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
    model = VGG16(num_classes=3)
    model.load_state_dict(torch.load('weights/vgg16_covid19.pth'))
    test_accuracy(model)