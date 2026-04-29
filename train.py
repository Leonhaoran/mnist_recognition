import torch
from torch import nn
from torch import optim

from model import MnistNetwork

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root='./mnist_images/train', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = MnistNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1} /10"
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "mnist.pth")
