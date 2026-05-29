import os
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.data import Dataset, Data, DataLoader
# from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch_geometric.utils import remove_self_loops
import numpy as np
import os.path as osp
import random
import sys
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import time, itertools
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score

from torch.utils.tensorboard import SummaryWriter

#
from torch.utils.data import Subset

pyOfFiles = open(r'D:\Dataset\AtCoder\python-ast.txt', "r", encoding="utf-8")
pyfilenameArray = pyOfFiles.readlines()
pyOfAst = open(r'D:\Dataset\AtCoder\python-ast.json', "r", encoding="utf-8")
pyastArray = pyOfAst.readlines()

javaOfFiles = open(r'D:\Dataset\CodeJame\java-ast.txt', "r", encoding="utf-8")
javafilenameArray = javaOfFiles.readlines()
javaOfAst = open(r'D:\Dataset\CodeJame\java-ast.json', "r", encoding="utf-8")
javaastArray = javaOfAst.readlines()

cOfFiles = open(r'D:\Dataset\CodeJame\c-ast.txt', "r", encoding="utf-8")
cfilenameArray = cOfFiles.readlines()
cOfAst = open(r'D:\Dataset\CodeJame\c-ast.json', "r", encoding="utf-8")
castArray = cOfAst.readlines()

cppOfFiles = open(r'D:\Dataset\CodeJame\cpp-ast.txt', "r", encoding="utf-8")
cppfilenameArray = cppOfFiles.readlines()
cppOfAst = open(r'D:\Dataset\CodeJame\cpp-ast.json', "r", encoding="utf-8")
cppastArray = cppOfAst.readlines()

def graphToMatrices(filename):
    # move to global to i`mprove performance
    from matrixFormation import adjacencyMatrixCreator, oneHotEncoder, High_latitudestensor
    if filename.split(".")[1] == "py":
        languageType = "python"
        filenameArray = pyfilenameArray
        astArray = pyastArray

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    elif filename.split(".")[1] == "java":
        languageType = "java"
        # 获取目标文件的路径
        filenameArray = javafilenameArray
        astArray = javaastArray

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]

    elif filename.split(".")[1] == "cpp":
        languageType = "c++"

        filenameArray = cppfilenameArray
        astArray = cppastArray

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    else:
        languageType = "c"

        filenameArray = cfilenameArray
        astArray = castArray

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]

    encodedMatrix = oneHotEncoder(astOfProgram, languageType)
    # encodedMatrix = encodeWithCodeBERT(astOfProgram)
    grapthTensor = High_latitudestensor(astOfProgram)
    '''
    2024/11/11 16点58分 
    adjacencyMatrix 改成为具有4个边类型的高维Tensor
        1、先使用python-java测试
        2、先用原模型测试新配对
            数据集100（先进行微调，调整代码使代码能够运行)
            数据集3W（数据集是否能使用）
            数据集6W（与原模型比对）
    '''
    return grapthTensor, encodedMatrix


def getTrainingPairs():
    trainingPairs = []
    ## read from trainPairs.txt (Java, py)
    ## read from txt (Java, py)
    listOfClones = open('CloneDetectionSrc/Codeccpp.txt', 'r')
    listOfNonClones = open('CloneDetectionSrc/nCodeccpp.txt', 'r')

    trainingPairs = listOfClones.readlines()
    nonCloneTrainingPairs = listOfNonClones.readlines()

    return trainingPairs, nonCloneTrainingPairs


class PairData(Data):
    def __inc__(self, key, value):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if 'index' in key or 'face' in key:
            return 1
        else:
            return 0

    # will return dataset pairs =>
    # ------------------------------------
    # data_point -> ASTAdjacencyMatrices + encodedMatrices  + Label (pair or not)
    # ------------------------------------


## Define n
n = 150000


# n=4405
# n=60
class TrainLoadData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None) -> object:
        super(TrainLoadData, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['CloneDetectionSrc/JNonClonePairs.txt']

    @property
    def processed_file_names(self):
        return ['1/data_{}.pt'.format(i) for i in range(n)]

    def process(self):
        # 获取所有的 pairs - self.raw_paths
        # 检查是否为 pair 并将其保存在数据中
        clonePairs, nonClonePairs = getTrainingPairs()

        if (len(clonePairs) > int(n / 2)):
            clonePairs = clonePairs[:(int(n / 2))]
        if (len(nonClonePairs) > int(n / 2)):
            nonClonePairs = nonClonePairs[:(int(n / 2))]

        i = 0

        # 处理克隆对
        for pairs in clonePairs:
            edgex1, encode1 = graphToMatrices(pairs.split(",")[0])
            edgex2, encode2 = graphToMatrices(pairs.split(",")[1][:-1])

            '''
                改1，先返回4个图，再构建一个高维Tensor再保存
            '''
            listForLabel = [1]
            labelTensor = torch.Tensor(listForLabel)
            matrix1 = torch.cat([torch.tensor(edgex1['ast_matric']).float(),
                                 torch.tensor(edgex1['cfg_matric']).float(),
                                 torch.tensor(edgex1['dfg_matric']).float(),
                                 torch.tensor(edgex1['ncs_matric']).float()], dim=1)

            matrix2 = torch.cat([torch.tensor(edgex2['ast_matric']).float(),
                                 torch.tensor(edgex2['cfg_matric']).float(),
                                 torch.tensor(edgex2['dfg_matric']).float(),
                                 torch.tensor(edgex2['ncs_matric']).float()], dim=1)

            data = PairData(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=matrix1,
                            edge_index2=matrix2, y=labelTensor)
            torch.save(data, osp.join(self.processed_dir, '1/data_{}.pt'.format(i)))
            i += 1

        # 处理非克隆对
        for pairs in nonClonePairs:
            edgex1, encode1 = graphToMatrices(pairs.split(",")[0])
            edgex2, encode2 = graphToMatrices(pairs.split(",")[1][:-1])

            listForLabel = [0]
            labelTensor = torch.Tensor(listForLabel)

            matrix1 = torch.cat([torch.tensor(edgex1['ast_matric']).float(),
                                 torch.tensor(edgex1['cfg_matric']).float(),
                                 torch.tensor(edgex1['dfg_matric']).float(),
                                 torch.tensor(edgex1['ncs_matric']).float()], dim=1)

            matrix2 = torch.cat([torch.tensor(edgex2['ast_matric']).float(),
                                 torch.tensor(edgex2['cfg_matric']).float(),
                                 torch.tensor(edgex2['dfg_matric']).float(),
                                 torch.tensor(edgex2['ncs_matric']).float()], dim=1)

            data = PairData(x1=torch.Tensor(encode1),
                            x2=torch.Tensor(encode2),
                            edge_index1=matrix1,
                            edge_index2=matrix2, y=labelTensor)
            torch.save(data, osp.join(self.processed_dir, '1/data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, '1/data_{}.pt'.format(idx)))
        return data


class MyTransform(object):
    def __call__(self, data):
        # Specify target - in our case its 0 only
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    sup_loss_all = 0
    unsup_loss_all = 0
    unsup_sup_loss_all = 0

    if use_unsup_loss:
        print("----Epoch training start----")
        for data, udata in zip(train_loader, unsup_train_loader):
            data = data.to(device)
            udata = udata.to(device)
            optimizer.zero_grad()
            criterion = BCEWithLogitsLoss()
            pred = model(data)
            sup_loss = criterion(pred, data.y)
            unsup_loss1 = model.unsup_loss1(udata, udata.x1_batch)  # unsup loss for java encoder
            unsup_loss2 = model.unsup_loss2(udata, udata.x2_batch)  # unsup loss for python encoder

            if separate_encoder:
                unsup_sup_loss1 = model.unsup_sup_loss1(udata, udata.x1_batch)
                unsup_sup_loss2 = model.unsup_sup_loss2(udata, udata.x2_batch)
                loss = sup_loss + (unsup_loss1 + unsup_loss2) + (unsup_sup_loss1 + unsup_sup_loss2) * lamda
            else:
                loss = sup_loss + (unsup_loss1 + unsup_loss2) * lamda

            loss.backward()

            sup_loss_all += sup_loss.item() * batch_size
            unsup_loss_all += (unsup_loss1.item() + unsup_loss2.item()) * batch_size
            if separate_encoder:
                unsup_sup_loss_all += (unsup_sup_loss1.item() + unsup_sup_loss2.item())

            loss_all += loss.item() * batch_size
            # print("Loss - %f "%loss_all)
            # print("sup_loss - %f "%sup_loss_all)
            # print("unsup_loss_all - %f "%unsup_loss_all)

            optimizer.step()
        print("----Epoch training over----")
        if separate_encoder:
            print(sup_loss_all, unsup_loss_all, unsup_sup_loss_all)
            return loss_all / len(train_loader.dataset), sup_loss_all, unsup_loss_all, unsup_sup_loss_all

        else:
            print(sup_loss_all / len(train_loader.dataset), unsup_loss_all / len(train_loader.dataset))
            return loss_all / len(train_loader.dataset), sup_loss_all / len(train_loader.dataset), unsup_loss_all / len(
                train_loader.dataset)

    else:
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            sup_loss = F.mse_loss(model(data), data.y)
            loss = sup_loss

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0
    precision = 0
    recall = 0
    accuracy = 0
    predictions = []
    grndTruth = []
    logits = []  # 用于存储模型的原始输出（logits）

    for data in loader:
        data = data.to(device)
        y_pred = torch.round(torch.sigmoid(model(data)))  # 二分类预测结果
        y_logit = model(data)  # 原始输出（logits）

        predictions.append(y_pred.cpu().detach().numpy().tolist())
        grndTruth.append(data.y.cpu().detach().numpy().tolist())
        logits.append(y_logit.cpu().detach().numpy().tolist())  # 收集 logits
        error += (y_pred - data.y).abs().sum().item()  # MAE

    # 将列表展平
    predictions = list(itertools.chain.from_iterable(predictions))
    grndTruth = list(itertools.chain.from_iterable(grndTruth))
    logits = list(itertools.chain.from_iterable(logits))

    # 计算准确率、精确率、召回率
    accuracy += accuracy_score(grndTruth, predictions)
    precision += precision_score(grndTruth, predictions)
    recall += recall_score(grndTruth, predictions)

    # 计算 ROC 曲线和 AUC 值
    roc_auc = roc_auc_score(grndTruth, logits)
    fpr, tpr, thresholds = roc_curve(grndTruth, logits)

    with open('last_epoch_roc_TenSorGCN.txt', 'w') as f:
        f.write('FPR,TPR\n')
        for i in range(len(fpr)):
            f.write(f'{fpr[i]},{tpr[i]}\n')

    return error / len(loader.dataset), accuracy, precision, recall, roc_auc


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from TensorGCNModel import Net


    # ============
    # Hyperparameters
    # ============
    target = 0
    dim = 64
    epochs = 25
    batch_size = 64
    lamda = 0.001
    use_unsup_loss = True
    separate_encoder = False

    tb = SummaryWriter()

    # # If transformation is required :
    # transform = T.Compose([MyTransform(), Complete()])
    # dataset = JavaClassificationDataset(root="./",transform=transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TrainLoadData(root="./")
    dataset = dataset.shuffle()

    # if dataset.data.x is None:
    #     max_degree = 0
    #     degs = []
    #     for data in dataset:
    #         degs += [degree(data.edge_index[0], dtype=torch.long)]
    #         max_degree = max(max_degree, degs[-1].max().item())
    #
    #     if max_degree < 1000:
    #         dataset.transform = T.OneHotDegree(max_degree)
    #     else:
    #         deg = torch.cat(degs, dim=0).to(torch.float)
    #         mean, std = deg.mean().item(), deg.std().item()
    #         dataset.transform = NormalizedDegree(mean, std)

    # Normalize targets to mean = 0 and std = 1.
    # mean = dataset.data.y[:, target].mean().item()
    # std = dataset.data.y[:, target].std().item()
    # dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    ####### Split datasets.
    '''
    trainSize = int(0.6 * len(dataset))
    valSize = int(0.2 * len(dataset))
    testSize = len(dataset) - trainSize - valSize
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize, testSize])
    '''
    # trainSize = int(0.6 * len(dataset))
    # valSize = int(0.2 * len(dataset))
    # testSize = len(dataset) - trainSize - valSize
    # train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize, testSize])

    # 泛化测试
    first_part_indices = list(range(0, 450))  # 0-449 (450条)
    second_part_indices = list(range(750, 1200))  # 750-1199 (450条)

    # 合并训练集索引 (总共900条)
    train_indices = first_part_indices + second_part_indices
    print(f"训练集总数: {len(train_indices)}条")

    # 创建训练集的子集
    train_dataset = Subset(dataset, train_indices)

    # 剩余部分的索引（所有不在训练集中的索引）
    all_indices = set(range(len(dataset)))  # 0-1499
    remaining_indices = list(all_indices - set(train_indices))
    remaining_size = len(remaining_indices)

    print(f"剩余数据量: {remaining_size}条")
    print(f"剩余数据范围: {min(remaining_indices)}-{max(remaining_indices)}")

    # 将剩余部分划分为验证集和测试集 (总共600条)
    val_size = 300  # 验证集大小300条
    test_size = 300  # 测试集大小300条

    # 检查剩余数据是否足够 (应该是600条，刚好足够)
    if remaining_size < val_size + test_size:
        print(f"警告: 剩余数据不足，剩余{remaining_size}条，需要{val_size + test_size}条")
        # 调整比例为剩余数据的一半
        val_size = remaining_size // 2
        test_size = remaining_size - val_size
        print(f"调整后: 验证集={val_size}条, 测试集={test_size}条")
    else:
        print(f"剩余数据充足: {remaining_size}条，刚好满足验证集{val_size}条 + 测试集{test_size}条")

    # 使用随机种子确保可重复性
    val_dataset, test_dataset = torch.utils.data.random_split(
        Subset(dataset, remaining_indices),
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_dataset, follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, follow_batch=['x1', 'x2'], batch_size=batch_size, shuffle=True)

    if use_unsup_loss:
        unsup_train_dataset = train_dataset
        unsup_train_loader = DataLoader(unsup_train_dataset, follow_batch=['x1', 'x2'], batch_size=batch_size,
                                        shuffle=True)
        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    dataset_num_features = 277

    model = Net(dataset_num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5,
                                                           min_lr=0.000001)
    best_val_error = None
    for epoch in range(1, epochs):
        start_time = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        print("Training epoch %d :" % epoch)
        if separate_encoder:
            loss, sup_loss, unsup_loss, unsup_sup_loss = train(epoch, use_unsup_loss)
        else:
            loss, sup_loss, unsup_loss = train(epoch, use_unsup_loss)

        print("Testing epoch %d :" % epoch)

        val_error, val_accuracy, val_prec, val_recall, val_roc_auc = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, test_accuracy, test_prec, test_recall, test_roc_auc = test(test_loader)
            best_val_error = val_error

        # 记录 ROC AUC 值
        tb.add_scalar('val_roc_auc', val_roc_auc, epoch)
        tb.add_scalar('test_roc_auc', test_roc_auc, epoch)

        end_time = time.time()
        epoch_duration = (end_time - start_time) / 3600

        # 将 ROC AUC 值写入日志文件
        with open('cloneDetection-EpochResults-meanTeacher.txt', 'a+') as f:
            if separate_encoder:
                f.write(
                    'Epoch: {:03d}, LR: {:7f}, T.Loss: {:.7f}, Sup-Loss: {:.7f},unSup-Loss: {:.7f},unSup-sup-Loss: {:.7f}, Validation MAE: {:.7f},Validation Acc: {:.7f},Validation Prec: {:.7f},Validation Rec: {:.7f},Validation ROC AUC: {:.7f},Test MAE: {:.7f},Test Acc: {:.7f},Test Prec: {:.7f},Test Rec: {:.7f},Test ROC AUC: {:.7f}, Time : {:.7f}'.format(
                        epoch, lr, loss, sup_loss, unsup_loss, unsup_sup_loss, val_error, val_accuracy, val_prec,
                        val_recall, val_roc_auc, test_error, test_accuracy, test_prec, test_recall, test_roc_auc,
                        epoch_duration))
            else:
                f.write(
                    'Epoch: {:03d}, LR: {:7f}, T.Loss: {:.7f}, Sup-Loss: {:.7f},unSup-Loss: {:.7f}, Validation MAE: {:.7f},Validation Acc: {:.7f},Validation Prec: {:.7f},Validation Rec: {:.7f},Validation ROC AUC: {:.7f},Test MAE: {:.7f},Test Acc: {:.7f},Test Prec: {:.7f},Test Rec: {:.7f},Test ROC AUC: {:.7f}, Time : {:.7f}'.format(
                        epoch, lr, loss, sup_loss, unsup_loss, val_error, val_accuracy, val_prec, val_recall,
                        val_roc_auc,
                        test_error, test_accuracy, test_prec, test_recall, test_roc_auc, epoch_duration))

            f.write('\n')

        # 保存模型
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   "cloneDetectionModels/Bi-meanTeacher_global_bigDB_global_avg_" + str(epoch) + ".pth")


    tb.close()
