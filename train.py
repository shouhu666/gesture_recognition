import  os
import numpy as np
import torch
# import torchvision.transforms as transforms
from model import Net
from load import MyDataset


path = '/home/sty16/deep_learning/gesture'
CORES = 4
batch = 16
lr = 0.001
epoch_num = 15
gpu = True


def adjust_learning_rate(optimizer, epoch, init_lr, step=80, decay=0.1):
    lr = init_lr * (decay ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    train_data = MyDataset(data_path=path, train=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=CORES)
    net = Net()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=lr, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0
    N = len(train_data.names)
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        net.cuda()
    for epoch in range(epoch_num):
        if epoch >= 1:
            adjust_learning_rate(optimizer, epoch, init_lr=lr, step=20, decay=0.1)
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            labels = labels.long()
            if gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(loss.item())
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        if epoch >= 0:
            filename = '%03i.pth' % (epoch+1)
            torch.save(net.state_dict(), filename)
            print('Saved:filename')