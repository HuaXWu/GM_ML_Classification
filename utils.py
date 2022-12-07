import os
import random
import matplotlib.pyplot as plt
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import sys
import numpy as np
import math
import json

"""
    utils
    author: wuhx
    data: 20221102
"""


# build warmup lr
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one_epoch_classify(model, optimizer, data_loader, device, epoch, lr_scheduler):
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        optimizer.zero_grad()
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        lr = optimizer.param_groups[0]["lr"]
        data_loader.desc = "[train epoch {}] loss: {:.8f}, acc: {:.8f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            lr
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, lr


best_acc = 0.


@torch.no_grad()
def evaluate_classify(model, data_loader, device, epoch, val_num, save_path):
    global best_acc
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    all_pred = []
    all_prob = []
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        probs = F.softmax(pred).detach().numpy()
        data = np.around(probs, 3)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        all_pred = all_pred + np.array(pred_classes).tolist()
        all_prob = all_prob + np.array(probs).tolist()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        val_accurate = accu_num / val_num
        # save best accuracy model
        # if val_accurate > best_acc:
        #     best_acc = val_accurate
        #     torch.save(model.state_dict(), save_path)
        data_loader.desc = "[valid epoch {}] loss: {:.8f}, acc: {:.8f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, all_pred, all_prob


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        #             raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = True
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label
