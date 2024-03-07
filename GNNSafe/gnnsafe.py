# import sys

import faiss
from argparse import Namespace

from torch.distributions import MultivariateNormal
from torch_geometric.utils import degree

from OutliersGenerate.utils import get_device
from backbone import *
import torch

# sys.path.append('..')
from OutliersGenerate.KNN import generate_outliers


class GNNSafe(nn.Module):
    """
    The model class of energy-based models for out-of-distribution detection
    The parameter args.use_reg and args.use_prop control the model versions:
        Energy: args.use_reg = False, args.use_prop = False
        Energy FT: args.use_reg = True, args.use_prop = False
        GNNSafe: args.use_reg = False, args.use_prop = True
        GNNSafe++ args.use_reg = True, args.use_prop = True
    """

    def __init__(self, d, c, args: Namespace):
        super(GNNSafe, self).__init__()
        if args.backbone == 'gcn':
            """
            Actor Datasets:
            in_channels=932
            hidden_channels=64
            out_channels=5
            num_layers=2
            dropout=0.0
            """
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

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        """return predicted logits"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        """
        Actor Datasets:
        x torch.Size([7600, 932])
        edge_index torch.Size([2, 30019])
        """
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
        if args.use_prop:  # use energy belief propagation
            neg_energy = self.propagation(neg_energy, edge_index, args.K, args.alpha)
        return neg_energy[node_idx]

    def loss_compute(self, dataset_ind, dataset_ood, criterion, device, args):
        """return loss for training"""
        """
        Actor Datasets:
        dataset_ind:
        Data(x=[7600, 932], edge_index=[2, 30019], y=[7600], 
            train_mask=[7600, 10], val_mask=[7600, 10], test_mask=[7600, 10], 
            node_idx=[7600])
        
        dataset_ood_tr
        Data(x=[7600, 932], edge_index=[2, 21146], y=[7600], node_idx=[7600])
        
        dataset_ood_te
        Data(x=[7600, 932], edge_index=[2, 21106], y=[7600], node_idx=[7600])
        
        # ind dataset actor: all nodes 7600 | centered nodes 7600 | edges 30019 | classes 5 | feats 932
        # ood tr dataset actor: all nodes 7600 | centered nodes 7600 | edges 21146
        """
        x_in, edge_index_in = dataset_ind.x.to(device), dataset_ind.edge_index.to(device)
        x_out, edge_index_out = dataset_ood.x.to(device), dataset_ood.edge_index.to(device)
        """
        x_in torch.Size([7600, 932])
        edge_index_in torch.Size([2, 30019])
        x_out torch.Size([7600, 932])
        edge_index_out torch.Size([2, 21146])
        """

        logits_in = self.encoder(x_in, edge_index_in)
        """
        x_in torch.Size([7600, 932]) edge_index_in torch.Size([2, 30019])
        logits_in torch.Size([7600, 5])
        """

        logits_out = self.encoder(x_out, edge_index_out)
        """
        x_out torch.Size([7600, 932]) edge_index_out torch.Size([2, 21146])
        logits_out torch.Size([7600, 5])
        """

        if args.generate_ood:

            device = get_device(args)
            res = faiss.StandardGpuResources()
            # penultimate_dim为数据集的features数
            # Actor 的特征数目为932
            penultimate_dim = dataset_ind.x.shape[1]
            KNN_index = faiss.GpuIndexFlatL2(res, penultimate_dim)
            ID = x_in

            # Standard Gaussian distribution
            # 创建一个高斯分布然后进行随机采样
            new_dis = MultivariateNormal(torch.zeros(penultimate_dim, device=device),
                                         torch.eye(penultimate_dim, device=device))
            # @issue 先固定采样600个
            negative_samples = new_dis.rsample((600,))
            sample_point = generate_outliers(
                # 分布内数据 x_in torch.Size([7600, 932])
                ID,
                # Faiss 向量搜索
                input_index=KNN_index,
                # 进行随机采样得到负样本 torch.Size([600, 932])
                negative_samples=negative_samples,
                # 从选择的分布内数据抽取的边界样本 2
                ID_points_num=2,
                # KNN距离 300
                K=300,
                # 多少ID样本用来定义边界 200
                select=200,
                # 采样范围0.1
                cov_mat=0.1,
                # 采样率 1
                sampling_ratio=1.0,
                # 挑选用于生成异常值的ID样本数
                pic_nums=2,
                # 特征的纬度 932
                depth=penultimate_dim,
                # 设备
                device=device
            )
            sample_point_label = torch.zeros(sample_point.shape[0], dtype=torch.long, device=device).view(-1)
            sample_point_edge = torch.randint(
                low=0,
                high=len(sample_point_label) - 1,
                size=(2, len(sample_point_label)),
                device=device
            )
            sample_point_logits_out = self.encoder(sample_point, sample_point_edge)
            sample_point_out = F.log_softmax(sample_point_logits_out, dim=1)
            sample_sup_loss = criterion(sample_point_out, sample_point_label)
        else:
            sample_sup_loss = 0

        train_in_idx, train_ood_idx = dataset_ind.splits['train'], dataset_ood.node_idx
        # train_in_idx torch.Size([760])
        # train_ood_idx torch.Size([7600]

        # compute supervised training loss
        if args.dataset in ('proteins', 'ppi'):
            sup_loss = criterion(logits_in[train_in_idx], dataset_ind.y[train_in_idx].to(device).to(torch.float))
        else:
            """
            torch.Size([7600, 5])
            torch.Size([760])
            torch.Size([760, 5])
            """
            pred_in = F.log_softmax(logits_in[train_in_idx], dim=1)
            sup_loss = criterion(pred_in, dataset_ind.y[train_in_idx].squeeze(1).to(device))

        if args.use_reg:  # If you use energy regularization
            if args.dataset in ('proteins', 'ppi'):  # for multi-label binary classification
                logits_in = torch.stack([logits_in, torch.zeros_like(logits_in)], dim=2)
                logits_out = torch.stack([logits_out, torch.zeros_like(logits_out)], dim=2)
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1).sum(dim=1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1).sum(dim=1)
            else:  # for single-label multi-class classification
                energy_in = - args.T * torch.logsumexp(logits_in / args.T, dim=-1)
                energy_out = - args.T * torch.logsumexp(logits_out / args.T, dim=-1)

            if args.use_prop:  # use energy belief propagation
                energy_in = self.propagation(energy_in, edge_index_in, args.K, args.alpha)[train_in_idx]
                energy_out = self.propagation(energy_out, edge_index_out, args.K, args.alpha)[train_ood_idx]
            else:
                energy_in = energy_in[train_in_idx]
                energy_out = energy_out[train_in_idx]

            # truncate to have the same length
            if energy_in.shape[0] != energy_out.shape[0]:
                min_n = min(energy_in.shape[0], energy_out.shape[0])
                energy_in = energy_in[:min_n]
                energy_out = energy_out[:min_n]

            # compute regularization loss
            reg_loss = torch.mean(F.relu(energy_in - args.m_in) ** 2 + F.relu(args.m_out - energy_out) ** 2)

            loss = sup_loss + args.lamda * reg_loss
        else:
            loss = sup_loss

        # print("loss", loss)
        # print("sample_point_loss", sample_sup_loss)
        # 加入采样的loss
        if args.generate_ood:
            loss += 0.001 * sample_sup_loss
        return loss
