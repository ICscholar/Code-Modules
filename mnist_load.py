import numpy as np
import struct
import gzip
import os

def load_mnist_data_gz(path):
    """
    从本地压缩的 IDX 文件 (gz) 加载 MNIST 数据。
    """
    def load_images_gz(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows, cols)
            return images

    def load_labels_gz(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    train_images_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')

    X_train = load_images_gz(train_images_path)
    y_train = load_labels_gz(train_labels_path)
    X_test = load_images_gz(test_images_path)
    y_test = load_labels_gz(test_labels_path)

    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, y_train, X_test, y_test

# 加载数据
X_train_seq, y_train, X_test_seq, y_test = load_mnist_data_gz('MNIST/')