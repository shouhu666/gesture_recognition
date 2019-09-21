import  os
import numpy as np
import torch
# import torchvision.transforms as transforms
from model import Net
from load import MyDataset


path = '/home/sty16/deep_learning/gesture'
CORES = 4
batch = 200
lr = 0.001
epoch_num = 15
gpu = True


def adjust_learning_rate(optimizer, epoch, init_lr, step=80, decay=0.1):
    lr = init_lr * (decay ** (epoch // step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    test_data = MyDataset(data_path=path, train=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=CORES)
    net = Net()
    pthfile = '014.pth'
    net.load_state_dict(torch.load(pthfile))

    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        net.cuda()
    net.eval()
    for i, (images, labels) in enumerate(test_loader):
        labels = labels.long()
        if gpu:
            images = images.cuda()
        outputs = net(images)
        outputs = np.array(outputs.cpu().detach().numpy())
        pos = np.argmax(outputs, axis=1)
        num_right = 0
        for i in range(len(pos)):
            if pos[i] == labels[i]:
                num_right = num_right + 1
        right = num_right / batch
        print(right)