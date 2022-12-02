import os.path

import torch
import torchvision
from torch.utils import model_zoo, data, tensorboard

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dense_modal():
    model = torchvision.models.densenet121(num_classes=1, pretrained=False).to(device)
    model.classifier = torch.nn.Sequential(model.classifier, torch.nn.Sigmoid())
    pretrained_dict = {k: v for k, v in torch.load("densenet121-a639ec97.pth").items() if k in model.state_dict() and 'classifier' not in k}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(model.state_dict())
    return model


def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(os.getcwd(), 'breast-simple'), transform=transform, )
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(os.getcwd(), 'breast-simple'), transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader


def train():
    ###################
    #     训练模型     #
    ###################
    print("-" * 59)
    print("\t\t\t模型开始进行训练")
    print("-" * 59)
    train_loader, _ = get_dataset()
    resnet_modal = get_dense_modal()
    optimizer = torch.optim.Adam(resnet_modal.classifier.parameters())
    cross_entropy_loss = torch.nn.BCELoss()
    resnet_modal.train()
    summary_writer = tensorboard.SummaryWriter()
    for epoch in range(50):
        for batch, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = resnet_modal(x_train)
            loss = cross_entropy_loss(
                output, torch.unsqueeze(y_train / 1.0, 1))
            loss.backward()
            optimizer.step()
            summary_writer.add_scalar('loss', loss.item(), epoch)
            if batch % 10 == 1:
                # summary_writer.add_image("Epoch: {}".format(epoch), x_train)
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch +
                      1, 50, batch + 1, len(train_loader), loss.item()))
    torch.save(resnet_modal.state_dict(), 'dense.pkl')


def eval():
    ######################
    #       验证模型      #
    ######################
    print("-" * 59)
    print("\t\t\t模型开始进行预测")
    print("-" * 59)
    _, test_loader = get_dataset()
    resnet_modal = get_dense_modal()
    resnet_modal.load_state_dict(torch.load("dense.pkl"))
    resnet_modal.eval()
    for images, labels in test_loader:
        image = images.to(device)
        label = labels.to(device)
        label = torch.unsqueeze(label, 1)
        outputs = resnet_modal(image)
        print(outputs)
        print(label)
        print('--'*10)
        predicted = torch.where(outputs.data > 0.5, 1, 0)
        # print("准确率:", (predicted == label).sum().item() / label.size()[0])


if __name__ == '__main__':
    eval()
