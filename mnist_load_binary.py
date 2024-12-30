import gzip
import struct
import os
import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data_gz(path):
    """
    Load MNIST data from local compressed IDX files (gz).
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

    # Binarize the images: threshold at 0.5
    X_train = (X_train / 255.0) > 0.5
    X_test = (X_test / 255.0) > 0.5

    return X_train.astype(float), y_train, X_test.astype(float), y_test


if __name__ == "__main__":
    # 假设MNIST数据集所在的路径，你需要根据实际情况修改这个路径
    data_path = 'MNIST/'
    X_train, y_train, X_test, y_test = load_mnist_data_gz(data_path)

    # 选择查看训练集中的前五张图片进行验证
    num_images_to_show = 5
    for i in range(num_images_to_show):
        # 获取当前图片数据并重塑为合适的二维形状用于可视化（原始形状可能是 (28, 28, 1)，这里去掉通道维度）
        image_data = X_train[i].reshape(28, 28)
        label = y_train[i]

        plt.subplot(1, num_images_to_show, i + 1)
        plt.imshow(image_data, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.show()
    
