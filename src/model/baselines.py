import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv

from src.common.parse import Arguments
from src.model.backbone import GCN, MLP, GAT, SGC, APPNP_Net, MixHop, GCNJK, GATJK
from src.outlier.energy import energy_propagation
from src.outlier.loss import compute_loss


class MSP(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(MSP, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        # return self.classifier(penultimate[node_idx], dataset.x, dataset.edge_index, node_idx).squeeze()

        logits, penultimate = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))
        logits, penultimate = logits[node_idx], penultimate[node_idx]
        sp = torch.softmax(logits, dim=-1)
        return sp.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class OE(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(OE, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(num_features, args.hidden_channels, num_classes, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        # return self.classifier(penultimate[node_idx]).squeeze()
        logits, penultimate = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))
        logits, penultimate = logits[node_idx], penultimate[node_idx]
        return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class ODIN(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(ODIN, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(num_features, args.hidden_channels, num_classes, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        odin_score = self.ODIN(dataset, node_idx, device, 1.0, 0.)
        return torch.Tensor(-np.max(odin_score, 1))

    def ODIN(self, dataset, node_idx, device, temper, noise_magnitude):
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        data = dataset.x
        data = Variable(data, requires_grad=True)
        edge_index = dataset.edge_index
        outputs, penultimate = self.encoder(data, edge_index)
        outputs, penultimate = outputs[node_idx], penultimate[node_idx]
        criterion = nn.CrossEntropyLoss()

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        # Using temperature scaling
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp)).to(device=device)
        loss = criterion(outputs, labels)

        datagrad = autograd.grad(loss, data)[0]
        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(datagrad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(input=data.data, other=-noise_magnitude, out=gradient)
        # tempInputs = torch.add(data.data, -noiseMagnitude1, gradient)

        outputs, penultimate = self.encoder(Variable(tempInputs), edge_index)
        outputs, penultimate = outputs[node_idx], penultimate[node_idx]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        return nnOutputs

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class Mahalanobis(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(Mahalanobis, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(num_features, args.hidden_channels, num_classes, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

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
                                              count - 1, 0.)
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
        for i in range(len(target)):
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

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class MaxLogits(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(MaxLogits, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'appnp':
            self.encoder = APPNP_Net(num_features, args.hidden_channels, num_classes, dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        # return self.classifier(penultimate[node_idx]).squeeze()

        logits, penultimate = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))
        logits, penultimate = logits[node_idx], penultimate[node_idx]
        return logits.max(dim=1)[0]

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class EnergyModel(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(EnergyModel, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=num_features, out_channels=num_classes, hops=2)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        # return self.classifier(penultimate[node_idx]).squeeze()

        logits, penultimate = self.encoder(dataset.x.to(device), dataset.edge_index.to(device))
        logits, penultimate = logits[node_idx], penultimate[node_idx]

        neg_energy = args.T * torch.logsumexp(logits / args.T, dim=-1)
        return neg_energy

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class EnergyProp(nn.Module):
    def __init__(self, num_features, num_classes, args: Arguments):
        super(EnergyProp, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(in_channels=num_features,
                               hidden_channels=args.hidden_channels,
                               out_channels=num_classes,
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'sgc':
            self.encoder = SGC(in_channels=num_features, out_channels=num_classes, hops=2)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers,
                               dropout=args.dropout, use_bn=True, heads=8, out_heads=1)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):

        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        logits, penultimate = logits[node_idx], penultimate[node_idx]

        neg_energy = 1.0 * torch.logsumexp(logits / 1.0, dim=-1)
        neg_energy_prop = energy_propagation(neg_energy, edge_index)
        return neg_energy_prop[node_idx]

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class GNNSafe(nn.Module):
    """
    The model class of energy-based models for out-of-distribution detection
    """

    def __init__(self, num_features, num_classes, args: Arguments):
        super(GNNSafe, self).__init__()
        if args.backbone == 'gcn':
            self.encoder = GCN(
                in_channels=num_features,
                hidden_channels=args.hidden_channels,
                out_channels=num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
                use_bn=True
            )
        elif args.backbone == 'mlp':
            self.encoder = MLP(in_channels=num_features, hidden_channels=args.hidden_channels,
                               out_channels=num_classes, num_layers=args.num_layers,
                               dropout=args.dropout)
        elif args.backbone == 'gat':
            self.encoder = GAT(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout,
                               use_bn=True)
        elif args.backbone == 'mixhop':
            self.encoder = MixHop(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gcnjk':
            self.encoder = GCNJK(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        elif args.backbone == 'gatjk':
            self.encoder = GATJK(num_features, args.hidden_channels, num_classes, num_layers=args.num_layers, dropout=args.dropout)
        else:
            raise NotImplementedError
        self.classifier = Classifier(in_features=args.hidden_channels)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, dataset, device):
        """return predicted logits"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        return logits

    def detect(self, dataset, node_idx, device, args):
        """return negative energy, a vector for all input nodes"""
        x, edge_index = dataset.x.to(device), dataset.edge_index.to(device)
        logits, penultimate = self.encoder(x, edge_index)
        # logits, penultimate = logits[node_idx], penultimate[node_idx]
        neg_energy = 1.0 * torch.logsumexp(logits / 1.0, dim=-1)
        # parser.add_argument('--K', type=int, default=2, help='number of layers for energy belief propagation')
        K = 2
        # parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual connection in propagation')
        alpha = 0.5
        if args.use_energy_propagation:
            neg_energy = energy_propagation(neg_energy, edge_index, num_prop_layers=K, alpha=alpha)
        return neg_energy[node_idx]
        # logits, penultimate = self.encoder(dataset.x, dataset.edge_index)
        # return self.classifier(penultimate[node_idx], dataset.x, dataset.edge_index, node_idx).squeeze()

    def loss_compute(self, dataset_ind: Data, dataset_ood: Data, synthesis_ood_dataset: Data, criterion, device, args):
        return compute_loss(dataset_ind, dataset_ood, synthesis_ood_dataset, self.encoder, self.classifier, criterion, device, args)


class GCNEncoder(nn.Module):
    """
    An encoder for feature extraction.
    """

    def __init__(self, in_features, in_channels):
        """
        Initialize a. encoder for feature extraction.
        Args:
            in_features: number of input features
            in_channels: number of input channels
        """
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels=in_channels, out_channels=in_channels // 2),
            GCNConv(in_channels=in_channels // 2, out_channels=in_channels // 4),
            GCNConv(in_channels=in_channels // 4, out_channels=in_features),
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        return self.log_softmax(x)


class Classifier(nn.Module):
    """
    A classifier for encoder penultimate output.
    """

    def __init__(self, in_features):
        """
        Initialize a classifier for encoder output.
        Args:
            in_features: number of input features
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=in_features // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features // 2, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, logit, x, edge_index, mask):
        return self.classifier(logit).squeeze().view(-1)
