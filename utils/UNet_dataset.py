import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    # 初始化函数
    def __init__(self, data_path):
        # 初始化函数，读取所有的data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    # 数据增强函数
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    # 数据获取函数
    # function:将图片读取，并处理成单通道图片。
    # 同时，因为 label 的图片像素点是0和255，因此需要除以255，变成0和1。
    # 同时，随机进行了数据增强。
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将数据转换为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转灰度
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # 处理标签，将像素为255的改成1
        if label.max() > 1:
            label = label / 255
        # 随机进行数据增强，为2的时候不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    #function:返回数据的大小
    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("../data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset=isbi_dataset,
        batch_size=2,
        shuffle=True    # Shuffle : 是否打乱数据位置，当为Ture时打乱数据，全部抛出数据后再次dataloader时重新打乱。
    )
    for image, label in train_loader:
        print(image.shape)







