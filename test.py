from model import MnistNetwork
from torchvision import transforms
from torchvision import datasets
import torch

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root="./mnist_images/test", transform=transform)
    model = MnistNetwork()
    model.load_state_dict(torch.load("mnist.pth"))

    correct = 0
    for batch_idx, (data, label) in enumerate(test_dataset):
        output = model(data)
        predict = output.argmax(1).item()

        if predict == label:
            correct += 1
        else:
            img_path = test_dataset.samples[batch_idx][0]
            print(f"wrong case: predict = {predict} label = {label} img_path = {img_path}")

    print(f"test accuracy = {correct / len(test_dataset)}")
