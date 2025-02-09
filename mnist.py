from torch.utils.data import Dataset, DataLoader
from keras.datasets import mnist
import torch as t
import os
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = self.get_cached_mnist_data()
        # Scale to [-1, 1]
        train_images = train_images.astype('float32') / 127.5 -1
        self.train_images = t.tensor(train_images, dtype=t.float32).unsqueeze(1)
        self.train_labels = train_labels = t.tensor(train_labels, dtype=t.long)

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):
        return self.train_images[idx], self.train_labels[idx]
    
    def get_cached_mnist_data(self):
        cache_dir = './mnist_cache'
        os.makedirs(cache_dir, exist_ok=True)
        train_images_path = os.path.join(cache_dir, 'train_images.npy')
        train_labels_path = os.path.join(cache_dir, 'train_labels.npy')
        test_images_path = os.path.join(cache_dir, 'test_images.npy')
        test_labels_path = os.path.join(cache_dir, 'test_labels.npy')

        if os.path.exists(train_images_path) and os.path.exists(train_labels_path) and \
            os.path.exists(test_images_path) and os.path.exists(test_labels_path):
            train_images = np.load(train_images_path)
            train_labels = np.load(train_labels_path)
            test_images = np.load(test_images_path)
            test_labels = np.load(test_labels_path)
        else:
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            np.save(train_images_path, train_images)
            np.save(train_labels_path, train_labels)
            np.save(test_images_path, test_images)
            np.save(test_labels_path, test_labels)

        return (train_images, train_labels), (test_images, test_labels)

    
def get_data_loader(batch_size):
    train_dataset = MNISTDataset()
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)






