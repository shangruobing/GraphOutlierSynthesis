from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from icecream import ic
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

from OutliersGenerate.KNN import generate_outliers
from OutliersGenerate.test import visualize, line
from backbone import GCN, MLP, GAT, SGC, APPNP_Net, MixHop, GCNJK, GATJK

count = 0


class MSP(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(MSP, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):
        # 是否需要把这个函数替换为分类器
        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1 - pred], dim=-1)
            max_sp = pred.max(dim=-1)[0]
            return max_sp.sum(dim=1)
        else:
            # return self.classifier(logits).view(-1)
            sp = torch.softmax(logits, dim=-1)
            return sp.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, criterion, device, args):
        train_idx = dataset_ind.train_mask
        x_in = dataset_ind.x.to(device)
        # 数据集的编码结果
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]
        # outputs = torch.randn(100, 128)
        # y = torch.randint(
        #     low=0,
        #     high=7,
        #     size=(len(outputs),),
        # )

        # visualize(logits_in, color=dataset_ind.y[train_idx], epoch=1)

        if args.generate_ood:
            # ic(x_in.size())
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=len(logits_in),
                # num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            # ic(x_in[0])
            # ic(sample_point[0])
            # 异常值的编码结果
            sample_point_logits_out = self.encoder(sample_point, sample_edge)
            # ic(logits_in.size())
            # ic(sample_point_logits_out.size())
            # 让分类器来对数据集和异常值的编码结果进行分类
            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            # ic(input_for_lr.size())
            # ic(input_for_lr )
            labels_for_lr = torch.cat([
                torch.ones(len(logits_in), device=device),
                torch.zeros(len(sample_point_logits_out), device=device)
            ])
            global count
            count += 1
            if count % 10 == 0:
                pass
                # line(x=input_for_lr.cpu().detach().numpy(), y=labels_for_lr.cpu().detach().numpy())
                # visualize(torch.cat([logits_in, sample_point_logits_out]), color=labels_for_lr, epoch=count)

            # ic(input_for_lr)
            # ic(torch.sigmoid(input_for_lr))
            criterion_BCE = nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            # line(x=input_for_lr, y=labels_for_lr)

            # ic(sample_sup_loss)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        # loss= encoder的loss + 分类器的loss
        if args.generate_ood:
            loss += sample_sup_loss
        return loss


class OE(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(OE, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1 - pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, criterion, device, args):

        train_idx = dataset_ind.train_mask

        x_in = dataset_ind.x.to(device)
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]
        # logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))
        # loss += 0.5 * -(logits_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()

        if args.generate_ood:
            loss += sample_sup_loss
        return loss


class ODIN(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(ODIN, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        odin_score = self.ODIN(dataset, node_idx, device, args.T, args.noise)
        return torch.Tensor(-np.max(odin_score, 1))

    def ODIN(self, dataset, node_idx, device, temper, noiseMagnitude):
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        data = dataset.x.to(device)
        data = Variable(data, requires_grad=True)
        edge_index = dataset.edge_index.to(device)
        outputs = self.encoder(data, edge_index)[node_idx]
        criterion = nn.CrossEntropyLoss()

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
        loss = criterion(outputs, labels)

        datagrad = autograd.grad(loss, data)[0]
        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        '''gradient[:, 0] = (gradient[:, 0]) / (63.0 / 255.0)
        gradient[:, 1] = (gradient[:, 1]) / (62.1 / 255.0)
        gradient[:, 2] = (gradient[:, 2]) / (66.7 / 255.0)'''
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))

        # Adding small perturbations to images
        tempInputs = torch.add(input=data.data, other=-noiseMagnitude, out=gradient)
        # tempInputs = torch.add(data.data, -noiseMagnitude1, gradient)

        outputs = self.encoder(Variable(tempInputs), edge_index)[node_idx]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs

    def loss_compute(self, dataset_ind: Data, criterion, device, args):

        train_idx = dataset_ind.train_mask
        x_in = dataset_ind.x.to(device)
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        if args.generate_ood:
            loss += sample_sup_loss
        return loss


class Mahalanobis(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(Mahalanobis, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, train_set, train_idx, test_set, node_idx, device, args):
        temp_list = self.encoder.feature_list(train_set.x.to(device), train_set.edge_index.to(device))[1]
        num_output = len(temp_list)
        feature_list = np.empty(num_output)
        count = 0
        for out in temp_list:
            feature_list[count] = out.size(1)
            count += 1
        num_classes = len(torch.unique(train_set.y))
        sample_mean, precision = self.sample_estimator(num_classes, feature_list, train_set, train_idx, device)
        in_score = self.get_Mahalanobis_score(test_set, node_idx, device, num_classes, sample_mean, precision,
                                              count - 1, args.noise)
        return torch.Tensor(in_score)

    def get_Mahalanobis_score(self, test_set, node_idx, device, num_classes, sample_mean, precision, layer_index,
                              magnitude):
        """
        Compute the proposed Mahalanobis confidence score on input dataset
        return: Mahalanobis score from layer_index
        """
        self.encoder.eval()
        Mahalanobis = []

        data, target = test_set.x.to(device), test_set.y[node_idx].to(device)
        edge_index = test_set.edge_index.to(device)
        data, target = Variable(data, requires_grad=True), Variable(target)

        out_features = self.encoder.intermediate_forward(data, edge_index, layer_index)[node_idx]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        datagrad = autograd.grad(loss, data)[0]

        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        '''gradient.index_copy_(1, torch.LongTensor([0]).to(device),
                     gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).to(device),
                     gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1 / 255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).to(device),
                     gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7 / 255.0))'''

        tempInputs = torch.add(input=data.data, other=-magnitude, out=gradient)
        # tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = self.encoder.intermediate_forward(tempInputs, edge_index, layer_index)[node_idx]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())

        return np.asarray(Mahalanobis, dtype=np.float32)

    def sample_estimator(self, num_classes, feature_list, dataset, node_idx, device):
        """
        compute sample mean and precision (inverse of covariance)
        return: sample_class_mean: list of class mean
                 precision: list of precisions
        """
        import sklearn.covariance

        self.encoder.eval()
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        correct = 0
        num_output = len(feature_list)
        num_sample_per_class = np.empty(num_classes)
        num_sample_per_class.fill(0)
        list_features = []
        for i in range(num_output):
            temp_list = []
            for j in range(num_classes):
                temp_list.append(0)
            list_features.append(temp_list)

        total = len(node_idx)
        output, out_features = self.encoder.feature_list(dataset.x.to(device), dataset.edge_index.to(device))
        output = output[node_idx]

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        target = dataset.y[node_idx].to(device)
        equal_flag = pred.eq(target).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(total):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

        sample_class_mean = []
        out_count = 0
        for num_feature in feature_list:
            temp_list = torch.Tensor(num_classes, int(num_feature)).to(device)
            for j in range(num_classes):
                temp_list[j] = torch.mean(list_features[out_count][j], 0)
            sample_class_mean.append(temp_list)
            out_count += 1

        precision = []
        for k in range(num_output):
            X = 0
            for i in range(num_classes):
                if i == 0:
                    X = list_features[k][i] - sample_class_mean[k][i]
                else:
                    X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

            # find inverse
            group_lasso.fit(X.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().to(device)
            precision.append(temp_precision)

        # print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

        return sample_class_mean, precision

    def loss_compute(self, dataset_ind: Data, criterion, device, args):

        train_idx = dataset_ind.train_mask
        x_in = dataset_ind.x.to(device)
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        if args.generate_ood:
            loss += sample_sup_loss
        return loss


class MaxLogits(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(MaxLogits, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(d, args.hidden_channels, c, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            pred = torch.sigmoid(logits).unsqueeze(-1)
            pred = torch.cat([pred, 1 - pred], dim=-1)
            max_logits = pred.max(dim=-1)[0]
            return max_logits.sum(dim=1)
        else:
            return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, criterion, device, args):

        train_idx = dataset_ind.train_mask
        x_in = dataset_ind.x.to(device)
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        if args.generate_ood:
            loss += sample_sup_loss
        return loss


class EnergyModel(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(EnergyModel, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=d, out_channels=c, hops=args.hops)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def detect(self, dataset, node_idx, device, args):

        logits = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))[node_idx]
        if args.dataset in ('proteins', 'ppi'):
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy

    def loss_compute(self, dataset_ind: Data, criterion, device, args):

        train_idx = dataset_ind.train_mask

        x_in = dataset_ind.x.to(device)
        logits_in = self.encoder(x_in, dataset_ind.edge_index.to(device))[train_idx]
        # logits_out = self.encoder(dataset_ood.x.to(device), dataset_ood.edge_index.to(device))[train_ood_idx]

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in, dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in, dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        '''if args.dataset in ('proteins', 'ppi'):
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
        else:
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
        if energy_in.shape[0] != energy_out.shape[0]:
            min_n = min(energy_in.shape[0], energy_out.shape[0])
            energy_in = energy_in[:min_n]
            energy_out = energy_out[:min_n]
        print(energy_in.mean().data, energy_out.mean().data)
        reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)
        # reg_loss = torch.mean(F.relu(energy_in - energy_out - args.m) ** 2)

        loss = sup_loss + args.lamda * reg_loss'''
        loss = sup_loss

        if args.generate_ood:
            loss += sample_sup_loss

        return loss


class EnergyProp(nn.Module):
    def __init__(self, d, c, args: Namespace):
        super(EnergyProp, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=d,
                               hidden_channels=args.hidden_channels,
                               out_channels=c,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=d, out_channels=c, hops=args.hops)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, l=1, alpha=0.5):
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(l):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):

        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'):
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        neg_energy_prop = self.propagation(neg_energy, edge_index)
        return neg_energy_prop[node_idx]

    def loss_compute(self, dataset_ind: Data, criterion, device, args):
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        # x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        train_idx = dataset_ind.train_mask

        logits_in = self.encoder(x_in, edge_index_in)[train_idx]
        # logits_out = self.encoder(x_out, edge_index_out)

        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_idx], dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        '''if args.dataset in ('proteins', 'ppi'):
            logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
            logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
        else:
            energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
            energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)
        energy_prop_in = self.propagation(energy_in, edge_index_in, args.prop_layers, args.alpha)[train_in_idx]
        energy_prop_out = self.propagation(energy_out, edge_index_out, args.prop_layers, args.alpha)[train_ood_idx]

        if energy_prop_in.shape[0] != energy_prop_out.shape[0]:
            min_n = min(energy_prop_in.shape[0], energy_prop_out.shape[0])
            energy_prop_in = energy_prop_in[:min_n]
            energy_prop_out = energy_prop_out[:min_n]
        print(energy_prop_in.mean().data, energy_prop_out.mean().data)
        reg_loss = torch.mean(F.relu(energy_prop_in - args.m_in) ** 2 + F.relu(args.m_out - energy_prop_out) ** 2)
        # reg_loss = torch.mean(F.relu(energy_prop_in - energy_prop_out - args.m) ** 2)

        loss = sup_loss + args.lamda * reg_loss'''
        loss = sup_loss

        if args.generate_ood:
            loss += sample_sup_loss

        return loss


class GNNSafe(nn.Module):
    """
    The model class of energy-based models for out-of-distribution detection
    """

    def __init__(self, d, c, args: Namespace):
        super(GNNSafe, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(
                in_channels=d,
                hidden_channels=args.hidden_channels,
                out_channels=c,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=args.use_bn
            )
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                               out_channels=c, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=args.use_bn)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = init_classifier(in_features=c)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        """return predicted logits"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        return self.encoder(x, edge_index)

    def propagation(self, e, edge_index, prop_layers=1, alpha=0.5):
        """energy belief propagation, return the energy after propagation"""
        e = e.unsqueeze(1)
        N = e.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm = 1. / d[col]
        value = torch.ones_like(row) * d_norm
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        for _ in range(prop_layers):
            e = e * alpha + matmul(adj, e) * (1 - alpha)
        return e.squeeze(1)

    def detect(self, dataset, node_idx, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits = self.encoder(x, edge_index)
        if args.dataset in ('proteins', 'ppi'):  # for multi-label binary classification
            logits = torch.stack([logits, torch.zeros_like(logits)], dim=2)
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1).sum(dim=1)
        else:  # for single-label multi-class classification
            neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        # if args.use_prop:  # use energy belief propagation
        #     neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind: Data, criterion, device, args):
        """return loss for training"""
        train_idx = dataset_ind.train_mask

        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        # x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)

        logits_in = self.encoder(x_in, edge_index_in)[train_idx]

        # logits_out = self.encoder(x_out, edge_index_out)
        if args.generate_ood:
            sample_point, sample_edge, sample_label = generate_outliers(
                x_in,
                device=device,
                num_nodes=dataset_ind.num_nodes,
                num_features=dataset_ind.num_features,
                num_edges=dataset_ind.num_edges,
            )
            sample_point_logits_out = self.encoder(sample_point, sample_edge)

            input_for_lr = self.classifier(torch.cat([logits_in, sample_point_logits_out])).squeeze()
            labels_for_lr = torch.cat([torch.ones(len(logits_in), device=device),
                                       torch.zeros(len(sample_point_logits_out), device=device)])
            criterion_BCE = torch.nn.BCEWithLogitsLoss()
            sample_sup_loss = criterion_BCE(input_for_lr, labels_for_lr)
            sample_sup_loss.backward(retain_graph=True)
        else:
            sample_sup_loss = 0

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_idx], dataset_ind.y[train_idx].to(device).to(torch.float))
        else:
            pred_in = F.log_softmax(logits_in[train_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_idx].squeeze(1).to(device))

        loss = sup_loss

        if args.generate_ood:
            loss += sample_sup_loss

        return loss


def init_classifier(in_features: int):
    """
    Initialize a classifier for encoder output.
    Args:
        in_features: number of input features

    Returns:

    """
    return nn.Sequential(
        nn.Linear(in_features=in_features, out_features=32, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=32, out_features=16, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=16, out_features=1, bias=True),
        nn.Sigmoid()
    )


class Classifier(nn.Module):
    def __init__(self, in_features):
        super(Classifier, self).__init__()
        self.classifier = init_classifier(in_features=in_features)

    def forward(self, x):
        return self.classifier(x)
