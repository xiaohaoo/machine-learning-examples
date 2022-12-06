import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from dn_cnn import DnCNN

from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    print("开始加载数据集......")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = Dataset(root="data/ct_simple/train/", transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True
    )
    dn_cnn = DnCNN(channels=3, num_of_layers=17).to(device)
    criterion = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(dn_cnn.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90], gamma=0.2
    )  # learning rates
    summary_writer = tensorboard.SummaryWriter()
    dn_cnn.train()
    mean = torch.as_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    for epoch in range(50):
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)
            dn_cnn.zero_grad()
            optimizer.zero_grad()
            out = dn_cnn(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            print("-> [epoch %d][%d] loss: %.4f" % (epoch + 1, i + 1, loss.item()))
            x = x * std + mean
            y = y * std + mean
            x = torch.round(x * 255)
            y = torch.round(y * 255)
            summary_writer.add_scalar("loss", loss.item(), epoch)
            summary_writer.add_images("Epoch {}".format(epoch), torch.cat((x, y), 0))
    torch.save(dn_cnn.state_dict(), "dn_cnn_net.pth")


def eval():
    print("开始验证模型......")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    test_dataset = Dataset(root="data/ct_simple/test", transform=transform)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=True
    )
    dn_cnn = DnCNN(channels=3, num_of_layers=17).to(device)
    dn_cnn.load_state_dict(torch.load("dn_cnn_net.pth"))
    dn_cnn.eval()
    mean = torch.as_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    std = torch.as_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
    summary_writer = tensorboard.SummaryWriter()
    for i, data in enumerate(test_loader):
        x, y = data[0].to(device), data[1].to(device)
        out = dn_cnn(x)
        x = x * std + mean
        out = out * std + mean
        x = torch.round(x * 255)
        out = torch.round(out * 255)
        summary_writer.add_images("Epoch {}".format(i), torch.cat((x, out), 0))
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(x.cpu())
        # plt.subplot(1, 3, 2)
        # plt.imshow(out.cpu())
        # plt.subplot(1, 3, 3)
        # plt.imshow(y)
        # plt.show()


if __name__ == "__main__":
    eval()
