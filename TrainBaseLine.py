import os
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.data import Dataset, Data, DataLoader
# from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch.nn import BCELoss, BCEWithLogitsLoss
from torch_geometric.utils import remove_self_loops

import os.path as osp
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import time, itertools
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
# pyOfFiles = open(r'D:\AST2\Complex\CodeJame\python-ast.txt', "r", encoding="utf-8")
# pyfilenameArray = pyOfFiles.readlines()
# pyOfAst = open(r'D:\AST2\Complex\CodeJame\python-ast.json', "r", encoding="utf-8")
# pyastArray = pyOfAst.readlines()
#
# javaOfFiles = open(r'D:\AST2\Complex\CodeJame\java-ast.txt', "r", encoding="utf-8")
# javafilenameArray = javaOfFiles.readlines()
# javaOfAst = open(r'D:\AST2\Complex\CodeJame\java-ast.json', "r", encoding="utf-8")
# javaastArray = javaOfAst.readlines()


#
from torch.utils.data import Subset
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# returns the encoding and adjacency matrix given the program filename
# def graphToMatrices(filename):
#     # move to global to i`mprove performance
#     from matrixFormation import oneHotEncoder, adjacencyMatrixCreator
#     languageType = ""
#     astOfProgram = []
#     if filename.split(".")[1] == "py":
#         languageType = "python"
#         listOfFiles = open('asts/python-ast.txt', "r", encoding="utf-8")
#         filenameArray = listOfFiles.readlines()
#         listOfAsts = open('asts/pnode.json', "r", encoding="utf-8")
#         astArray = listOfAsts.readlines()
#         filename += "\n"
#         idxOfFile = filenameArray.index(filename)
#         astOfProgram = astArray[idxOfFile]
#
#     elif filename.split(".")[1] == "java":
#         languageType = "java"
#         # 获取目标文件的路径
#         listOfFiles = open('asts/java-ast.txt', "r", encoding="utf-8")
#         filenameArray = listOfFiles.readlines()
#
#         # 根据文件所在索引获取node进行one-hotencoding
#         listOfnode = open('asts/jnode.json', "r", encoding="utf-8")
#         astArray = listOfnode.readlines()
#
#         filename += "\n"
#         idxOfFile = filenameArray.index(filename)
#         astOfProgram = astArray[idxOfFile]
#     else:
#         languageType = "c"
#         # 获取目标文件的路径
#         listOfFiles = open('asts/C-ast.txt', "r", encoding="utf-8")
#         filenameArray = listOfFiles.readlines()
#
#         # 根据文件所在索引获取node进行one-hotencoding
#         listOfnode = open('asts/Cnode.json', "r", encoding="utf-8")
#         astArray = listOfnode.readlines()
#
#         filename += "\n"
#         idxOfFile = filenameArray.index(filename)
#         astOfProgram = astArray[idxOfFile]
#
#     encodedMatrix = oneHotEncoder(astOfProgram, languageType)
#     adjacencyMatrix, num_nodes = adjacencyMatrixCreator(astOfProgram)
#     return adjacencyMatrix, encodedMatrix, num_nodes

def graphToMatrices(filename):
    # move to global to i`mprove performance
    from matrixFormation import adjacencyMatrixCreator, oneHotEncoder, High_latitudestensor
    languageType = ""
    astOfProgram = []
    if filename.split(".")[1] == "java":
        languageType = "java"
        # 获取目标文件的路径
        listOfFiles = open('asts/all_graphast/java.txt', "r", encoding="utf-8")
        filenameArray = listOfFiles.readlines()

        # 根据文件所在索引获取高维Tensor
        listOfAst = open('asts/all_graphast/java-ast.json', "r", encoding="utf-8")
        astArray = listOfAst.readlines()

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    elif filename.split(".")[1] == "c":
        languageType = "c"
        # 获取目标文件的路径
        listOfFiles = open('asts/c.txt', "r", encoding="utf-8")
        filenameArray = listOfFiles.readlines()
        # 根据文件所在索引获取高维Tensor
        listOfAst = open('asts/all_graphast/c-ast.json', "r", encoding="utf-8")
        astArray = listOfAst.readlines()

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    else:
        languageType = "cpp"
        # 获取目标文件的路径
        listOfFiles = open('asts/cpp.txt', "r", encoding="utf-8")
        filenameArray = listOfFiles.readlines()
        # 根据文件所在索引获取高维Tensor
        listOfAst = open('asts/all_graphast/cpp-ast.json', "r", encoding="utf-8")
        astArray = listOfAst.readlines()

        filename += "\n"
        idxOfFile = filenameArray.index(filename)
        astOfProgram = astArray[idxOfFile]
    encodedMatrix = oneHotEncoder(astOfProgram, languageType)
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
    listOfClones = open('CloneDetectionSrc/1.txt', 'r')
    listOfNonClones = open('CloneDetectionSrc/n1.txt', 'r')

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
n = 600

# n=4405
# n=60
class TrainLoadData(Dataset):
    def __init__(self, root, transform=None, pre_transform=None) -> object:
        super(TrainLoadData, self).__init__(root, transform, pre_transform)

        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['CloneDetectionSrc/fnonClonePairs.txt']

    @property
    def processed_file_names(self):
        return ['1/data_{}.pt'.format(i) for i in range(n)]

    def process(self):
        # get all the pairs - self.raw_paths
        # check pair or not and then save the pair in the data
        clonePairs, nonClonePairs = getTrainingPairs()

        if (len(clonePairs) > int(n / 2)):
            clonePairs = clonePairs[:(int(n / 2))]
        if (len(nonClonePairs) > int(n / 2)):
            nonClonePairs = nonClonePairs[:(int(n / 2))]

        i = 0
        for pairs in clonePairs:

            matrix1, encode1 = graphToMatrices(pairs.split(",")[0])
            matrix2, encode2 = graphToMatrices(pairs.split(",")[1][:-1])

            listForLabel = [1]
            labelTensor = torch.Tensor(listForLabel)
            # data = Data(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2),num_nodes1=num_nodes1,num_nodes2=num_nodes2 y=1)

            data = PairData(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2),
                           edge_index1=torch.Tensor(matrix1['ast_matric']),
                           edge_index2=torch.Tensor(matrix2['ast_matric']), y=labelTensor)
            # data1 = Data(x=torch.Tensor(encode1), edge_index=torch.LongTensor(matrix1), num_nodes=num_nodes1)
            # data2 = Data(x=torch.Tensor(encode2), edge_index=torch.LongTensor(matrix2), num_nodes=num_nodes2)
            # data = Data(data1=data1, data2=data2, y=1)
            torch.save(data, osp.join(self.processed_dir, '1/data_{}.pt'.format(i)))
            i += 1

        for pairs in nonClonePairs:
            matrix1, encode1 = graphToMatrices(pairs.split(",")[0])
            matrix2, encode2 = graphToMatrices(pairs.split(",")[1][:-1])
            listForLabel = [0]
            labelTensor = torch.Tensor(listForLabel)
            # data = Data(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1), edge_index2=torch.Tensor(matrix2),num_nodes1=num_nodes1,num_nodes2=num_nodes2 y=0)
            data = PairData(x1=torch.Tensor(encode1), x2=torch.Tensor(encode2), edge_index1=torch.Tensor(matrix1['ast_matric']),
                            edge_index2=torch.Tensor(matrix2['ast_matric']), y=labelTensor)
            # data1 = Data(x=torch.Tensor(encode1), edge_index=torch.LongTensor(matrix1), num_nodes=num_nodes1)
            # data2 = Data(x=torch.Tensor(encode2), edge_index=torch.LongTensor(matrix2), num_nodes=num_nodes2)
            # data = Data(data1=data1, data2=data2, y=0)
            torch.save(data, osp.join(self.processed_dir, '1/data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):

        data = torch.load(osp.join(self.processed_dir, '1/data_{}.pt'.format(idx)))
        # print(data)
        # print(data.edge_index2.shape)
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
    fpr, tpr, thresholds = roc_curve(grndTruth, logits)
    roc_auc = roc_auc_score(grndTruth, logits)

    with open('last_epoch_roc_Rubhus.txt', 'w') as f:
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
    from baseLineModel import Net

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

    dataset = TrainLoadData(root="./")
    dataset = dataset.shuffle()
    print('num_features : {}\n'.format(dataset.num_features))

    #
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

    ###### Split datasets.
    trainSize = int(0.6 * len(dataset))
    valSize = int(0.2 * len(dataset))
    testSize = len(dataset) - trainSize - valSize
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [trainSize, valSize, testSize])


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

    dataset_num_features = 212

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
