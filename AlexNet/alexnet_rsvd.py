import copy
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn as nn

from alexnet_train import AlexNet
from alexnet_test import model_compression_ratio, test_accuracy

class RSVDLinear(nn.Module):
    def __init__(self, original_layer, layer_name, rank):
        super(RSVDLinear, self).__init__()
        self.ranks = rank
        print(f"Layer rank for RSVD: {self.ranks}")

        W = original_layer.weight.data.cpu().numpy()
        bias = original_layer.bias.data.cpu().numpy() if original_layer.bias is not None else None

        U, S, Vt = randomized_svd(W, n_components=rank, n_iter=10, random_state=42)

        self.U = nn.Parameter(torch.from_numpy(U).float())
        self.S = nn.Parameter(torch.from_numpy(S).float())
        self.Vt = nn.Parameter(torch.from_numpy(Vt).float())

        if bias is not None:
            self.bias = nn.Parameter(torch.from_numpy(bias).float())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return ((x @ self.Vt.T) * self.S) @ self.U.T + self.bias

def compress_alexnet_rsvd(original_model):
    compressed_model = copy.deepcopy(original_model)
    
    rsvd_ranks = {'linear1': 100, 'linear2': 10}

    for layer_name, layer in original_model.classifier.named_children():
        if isinstance(layer, nn.Linear) and layer_name != "linear3":
            compressed_layer = RSVDLinear(layer, layer_name, rsvd_ranks[layer_name])
            setattr(compressed_model.classifier, layer_name, compressed_layer)
        else:
            setattr(compressed_model.classifier, layer_name, layer)

    return compressed_model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=10)
    model.load_state_dict(torch.load('weights/alexnet_cifar10_100epoch.pth'))

    rsvd_compressed_model = compress_alexnet_rsvd(model)
    rsvd_compressed_model = rsvd_compressed_model.to(device)
    torch.save(rsvd_compressed_model.state_dict(), "weights/rsvd_alexnet.pth")

    print("=" * 100, "\n",)
    print("Original model:")
    test_accuracy(model)
    print("RSVD-compressed model:")
    test_accuracy(rsvd_compressed_model)
    print("Compression:")
    model_compression_ratio(model, rsvd_compressed_model)