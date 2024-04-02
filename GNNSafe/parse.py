def parser_add_main_args(parser):
    # setup and protocol
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--ood_type', type=str, default='structure', choices=['structure', 'label', 'feature', 'knn'])
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)

    # model network
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--backbone', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for GNN classifiers')

    # training
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)

    # generate outliers
    parser.add_argument('--generate_ood', action='store_true')
