from torch.utils.data import DataLoader

from model import MnistNetwork
from torchvision import transforms, datasets
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root="./mnist_images/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MnistNetwork().to(device)
    model.load_state_dict(torch.load("mnist_gpu.pth", map_location=device))
    model.eval()

    correct = 0
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            pred = output.argmax(1).item()

            if pred == label:
                correct += 1
            else:
                img_path = test_dataset.samples[idx][0]
                print(f"wrong case: predict = {pred} label = {label.item()} img_path = {img_path}")

    print(f"test accuracy = {correct / len(test_dataset)}")
