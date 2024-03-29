from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
import numpy as np
import feather

"""
    read origin data
    author: wuhx
    data: 20221106
"""


def read_split_data():
    X = feather.read_dataframe("./data/T2D/X.feather")
    Y = feather.read_dataframe("./data/T2D/label.feather")
    # X = X.iloc[:, 0:-3]
    Y = Y["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y)
    print("X_train shape is {}, X_test shape is {}".format(X_train.shape, X_test.shape))
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_train = torch.from_numpy(np.array(X_train))
    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    X_test = torch.from_numpy(np.array(X_test))
    y_test = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)
    X_train = X_train.reshape(352, 6, 101)
    X_test = X_test.reshape(88, 6, 101)
    X_train = torch.unsqueeze(X_train, dim=1)
    X_test = torch.unsqueeze(X_test, dim=1)
    X_train = X_train.type(torch.float32)
    X_test = X_test.type(torch.float32)
    # step to tensor
    train_tensor = data.TensorDataset(X_train, y_train)
    test_tensor = data.TensorDataset(X_test, y_test)

    # step 5 to dataloader
    train_loader = data.DataLoader(train_tensor, batch_size=16, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(test_tensor, batch_size=16, shuffle=False, num_workers=8)
    return train_loader, val_loader
