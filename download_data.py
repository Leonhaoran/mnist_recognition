from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train_data = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_data = MNIST(root='./data', train=False, transform=ToTensor(), download=True)

from torchvision.transforms import ToPILImage

train_data = [(ToPILImage()(img), label) for img, label in train_data]
test_data = [(ToPILImage()(img), label) for img, label in test_data]

import os
import secrets
_

def save_images(dataset, folder_name):
    root_dir = os.path.join('./mnist_images', folder_name)

    if os.path.isdir(root_dir) and any(os.scandir(root_dir)):
        print(f"Directory '{root_dir}' already exists and is not empty. Skipping saving images.")
        return

    os.makedirs(root_dir, exist_ok=True)
    for i in range(len(dataset)):
        img, label = dataset[i]
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        random_filename = secrets.token_hex(8) + '.png'
        img.save(os.path.join(label_dir, random_filename))


save_images(train_data, 'train')
save_images(test_data, 'test')
