import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MnistDataset(Dataset):
    def __init__(self, csv_file="./mnist_train.csv"):
        super().__init__()
        self.data_df = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # 레이블 전처리
        label = self.data_df.iloc[index, 0]
        target = torch.zeros((10))
        target[label] = 1.0  # 원핫인코딩
        img = torch.FloatTensor(self.data_df.iloc[index, 1:].values) / 255.0  # 노말라이징
        return label, img, target

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation="none", cmap="Blues")
