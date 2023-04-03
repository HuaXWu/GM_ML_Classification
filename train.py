import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset

import loadData
from cnn_model import CNNModel
from utils import *

"""
    train function
    author: wuhx
    data: 20221120
"""
# transfer-learning save path
save_path = "./model_path/CNN_T2D_.pth"
"""
    load image data
"""
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(root="/root/autodl-tmp/origin_image/Hepatocirrhosis")
print(train_images_path)
print(val_images_path)

img_size = 50
data_transform = {
    "train": transforms.Compose([
        #                                 transforms.RandomResizedCrop(img_size),
        #                                  transforms.RandomHorizontalFlip(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.052470338, 0.052470338, 0.052470338], [0.16041122, 0.16041122, 0.16041122])
    ]),
    "val": transforms.Compose([
        #                                transforms.Resize(int(img_size * 1.143)),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.047967374, 0.047967374, 0.047967374], [0.1504331, 0.1504331, 0.1504331])
    ])}

train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=data_transform["train"])

val_dataset = MyDataSet(images_path=val_images_path,
                        images_class=val_images_label,
                        transform=data_transform["val"])

# val_num = len(val_dataset)

batch_size = 16
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=nw,
                                           collate_fn=train_dataset.collate_fn)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True,
                                         num_workers=nw,
                                         collate_fn=val_dataset.collate_fn)
"""
    load matrix data
"""
# train_loader, val_loader = loadData.read_split_data()
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(7500, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 2))

val_num = 26
# net = CNNModel(num_classes=2)
print(net)
params = [p for p in net.parameters() if p.requires_grad]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
optimizer = optim.SGD(params, lr=0.01)
lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 50,
                                   warmup=True, warmup_epochs=10)

if __name__ == '__main__':
    epochs = 50
    all_train_acc = np.zeros(epochs)
    all_train_loss = np.zeros(epochs)
    all_lr = np.zeros(epochs)

    all_test_acc = np.zeros(epochs)
    all_test_loss = np.zeros(epochs)

    all_pred = np.zeros(epochs)
    all_prob = np.zeros(epochs)
    best_acc = 0.
    best_pred = []
    best_prob = []
    for epoch in range(epochs):
        train_loss, train_acc, train_lr = train_one_epoch_classify(net, optimizer,
                                                                   train_loader,
                                                                   device,
                                                                   epoch,
                                                                   lr_scheduler)
        all_train_acc[epoch] = train_acc
        all_train_loss[epoch] = train_loss
        all_lr[epoch] = train_lr

        test_loss, test_acc, pred, prob = evaluate_classify(net,
                                                            val_loader,
                                                            device,
                                                            epoch,
                                                            val_num,
                                                            save_path)
        all_test_acc[epoch] = test_acc
        all_test_loss[epoch] = test_loss
        if test_acc > best_acc:
            best_acc = test_acc
            best_prob = prob
            best_pred = pred
    print("test_loss is{}\n test_acc is{}\n best_prob is{}\n best_pred is{}\n best_acc is {}\n ".format(
        ",".join(str(i) for i in all_test_loss),
        ",".join(str(i) for i in all_test_acc),
        best_prob,
        best_pred,
        best_acc
    ))
    all_test_acc.sort()
    print(all_test_acc)
