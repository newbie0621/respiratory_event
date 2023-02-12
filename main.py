import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from CNN import Mymodel
from sklearn.metrics import accuracy_score,precision_score,recall_score
import warnings
path = '附件/呼吸事件'


###构建自己的数据集
class Mydataset(Dataset):
    def __init__(self, path):
        super().__init__()
        data = np.load(path, allow_pickle=True)
        self.data = data

    def __getitem__(self, idx):
        time_series = self.data[0][idx]
        label = self.data[1][idx].astype(np.int)
        X = torch.tensor(time_series, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.int64)
        return X, y

    def __len__(self):
        return len(self.data[0])


###模型的训练
def train(model, writer, device, train_loader, optimizer, loss_fn, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_fn(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print('模型在训练集上的损失:{}'.format(train_loss))
    writer.add_scalar(tag='train_loss', scalar_value=train_loss, global_step=epoch)
    accuracy = train_correct / len(train_loader.dataset)
    writer.add_scalar(tag='train_acc', scalar_value=accuracy, global_step=epoch)
    print('模型在训练集上的预测精度:{}'.format(accuracy))


###模型的测试
def test(model, writer, device, test_loader, loss_fn, epoch, test_acc):
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    print('模型在测试集上的损失:{}'.format(test_loss))
    writer.add_scalar(tag='test_loss', scalar_value=test_loss, global_step=epoch)
    accuracy = test_correct / len(test_loader.dataset)
    writer.add_scalar(tag='test_acc', scalar_value=accuracy, global_step=epoch)
    print('模型在测试集上的预测精度:{}'.format(accuracy))
    if accuracy > test_acc:
        torch.save(model, './model/BEST_MODEL.pth')
    return accuracy


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    ###加载数据
    train_dataset = Mydataset(os.path.join(path, 'OSA-train-5people.npy'))
    test_dataset = Mydataset(os.path.join(path, 'OSA-test-3people.npy'))

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

    ###将过程结果写入tensorboard
    writer = SummaryWriter('logs')

    ###根据电脑配置选择GPU或者cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###神经网络的训练和测试
    model = Mymodel()
    model = model.to(device)

    ###随机梯度下降法
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ###优化学习率，防止模型不能收敛
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

    ###用交叉熵函数作为损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    EPOCH = 100
    test_acc = 0
    for epoch in range(EPOCH):
        print('EPOCH**************************{}/{}'.format(epoch + 1, EPOCH))
        train(model=model, writer=writer, device=device, train_loader=train_dataloader, optimizer=optimizer,
              loss_fn=loss_fn, epoch=epoch)
        test_acc = test(model=model, writer=writer, device=device, test_loader=test_dataloader, loss_fn=loss_fn,
                        epoch=epoch, test_acc=test_acc)
        scheduler.step()
    writer.close()

    ###计算准确率，精确率和召回率
    #首先加载测试数据
    dataset_test=np.load(os.path.join(path, 'OSA-test-3people.npy'),allow_pickle=True)
    X = torch.tensor(dataset_test[0], dtype=torch.float32)
    y_true = dataset_test[1]

    #加载模型
    best_model=torch.load('./model/BEST_MODEL.pth')

    #模型在测试集上的预测值
    y_pred=best_model(X).argmax(dim=1).numpy()
    print('测试结果：{}'.format(y_pred))


    #计算准确率，精确率和召回率
    accuracy=accuracy_score(y_true,y_pred)
    precision=precision_score(y_true,y_pred)
    recall=recall_score(y_true,y_pred)

    print('准确率：{}'.format(accuracy))
    print('精确率：{}'.format(precision))
    print('召回率：{}'.format(recall))