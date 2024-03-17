from collections import defaultdict

import torch


class ClassifyLogger:
    """ logger for node classification task, reporting train/valid/test accuracy or roc, auc for classification """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            # argmin = result[:, 3].argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'Highest Test: {result[:, 2].max():.2f}')
            print(f'Chosen epoch: {argmax + 1}')
            print(f'Final Train: {result[argmax, 0]:.2f}')
            print(f'Final Test: {result[argmax, 2]:.2f}')
            # self.test = result[argmax, 2]
        else:
            result = 100 * torch.tensor(self.results)
            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                test1 = r[:, 2].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test2 = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))
            best_result = torch.tensor(best_results)
            if best_result.shape[0] == 1:
                print(f'All runs:')
                r = best_result[:, 0]
                print(f'Highest Train: {r.mean():.2f}')
                r = best_result[:, 1]
                print(f'Highest Test: {r.mean():.2f}')
                r = best_result[:, 2]
                print(f'Highest Valid: {r.mean():.2f}')
                r = best_result[:, 3]
                print(f'  Final Train: {r.mean():.2f}')
                r = best_result[:, 4]
                print(f'   Final Test: {r.mean():.2f}')
            else:
                print(f'All runs:')
                r = best_result[:, 0]
                print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 1]
                print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 2]
                print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 3]
                print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 4]
                print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            return best_result[:, 4]

    def get_statistics(self):
        result = 100 * torch.tensor(self.results)
        best_results = []
        for r in result:
            train1 = r[:, 0].max().item()
            test1 = r[:, 2].max().item()
            valid = r[:, 1].max().item()
            train2 = r[r[:, 1].argmax(), 0].item()
            test2 = r[r[:, 1].argmax(), 2].item()
            best_results.append((train1, test1, valid, train2, test2))
        best_result = torch.tensor(best_results)
        return {
            "Highest Train": round(best_result[:, 0].mean().item(), 4),
            "Highest Test": round(best_result[:, 1].mean().item(), 4),
            "Highest Valid": round(best_result[:, 2].mean().item(), 4),
            "Final Train": round(best_result[:, 3].mean().item(), 4),
            "Final Test": round(best_result[:, 4].mean().item(), 4),
        }


class DetectLogger:
    """ logger for ood detection task, reporting test auroc/aupr/fpr95 for ood detection """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) % 3 == 2
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            ood_result, test_score, valid_loss = result[:, :-2], result[:, -2], result[:, -1]
            argmin = valid_loss.argmin().item()
            print(f'Run {run + 1:02d}:')
            print(f'Chosen epoch: {argmin + 1}')
            for k in range(result.shape[1] // 3):
                print(f'OOD Test {k + 1} Final AUROC: {ood_result[argmin, k * 3]:.2f}')
                print(f'OOD Test {k + 1} Final AUPR: {ood_result[argmin, k * 3 + 1]:.2f}')
                print(f'OOD Test {k + 1} Final FPR95: {ood_result[argmin, k * 3 + 2]:.2f}')
            print(f'IND Test Score: {test_score[argmin]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            ood_te_num = result.shape[2] // 3

            best_results = []
            for r in result:
                ood_result, test_score, valid_loss = r[:, :-2], r[:, -2], r[:, -1]
                score_val = test_score[valid_loss.argmin()].item()
                ood_result_val = []
                for k in range(ood_te_num):
                    auroc_val = ood_result[valid_loss.argmin(), k * 3].item()
                    aupr_val = ood_result[valid_loss.argmin(), k * 3 + 1].item()
                    fpr_val = ood_result[valid_loss.argmin(), k * 3 + 2].item()
                    ood_result_val += [auroc_val, aupr_val, fpr_val]
                best_results.append(ood_result_val + [score_val])

            best_result = torch.tensor(best_results)

            if best_result.shape[0] == 1:
                print(f'All runs:')
                for k in range(ood_te_num):
                    r = best_result[:, k * 3]
                    print(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f}')
                    r = best_result[:, k * 3 + 1]
                    print(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f}')
                    r = best_result[:, k * 3 + 2]
                    print(f'OOD Test {k + 1} Final FPR: {r.mean():.2f}')
                r = best_result[:, -1]
                print(f'IND Test Score: {r.mean():.2f}')
            else:
                print(f'All runs:')
                for k in range(ood_te_num):
                    r = best_result[:, k * 3]
                    print(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f}')
                    r = best_result[:, k * 3 + 1]
                    print(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ± {r.std():.2f}')
                    r = best_result[:, k * 3 + 2]
                    print(f'OOD Test {k + 1} Final FPR: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, -1]
                print(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}')

            return best_result

    def get_statistics(self):
        result = 100 * torch.tensor(self.results)
        ood_te_num = result.shape[2] // 3
        best_results = []
        for r in result:
            ood_result, test_score, valid_loss = r[:, :-2], r[:, -2], r[:, -1]
            score_val = test_score[valid_loss.argmin()].item()
            ood_result_val = []
            for k in range(ood_te_num):
                auroc_val = ood_result[valid_loss.argmin(), k * 3].item()
                aupr_val = ood_result[valid_loss.argmin(), k * 3 + 1].item()
                fpr_val = ood_result[valid_loss.argmin(), k * 3 + 2].item()
                ood_result_val += [auroc_val, aupr_val, fpr_val]
            best_results.append(ood_result_val + [score_val])
        best_result = torch.tensor(best_results)
        return {
            "AUROC": round(best_result[:, 0].mean().item(), 4),
            "AUPR": round(best_result[:, 1].mean().item(), 4),
            "FPR": round(best_result[:, 2].mean().item(), 4),
            "SCORE": round(best_result[:, 3].mean().item(), 4),
        }

# def save_result(results, args):
#     if args.dataset in ('cora', 'amazon-photo', 'coauthor-cs'):
#         filename = f'results/{args.dataset}-{args.ood_type}.csv'
#     else:
#         filename = f'results/{args.dataset}.csv'
#
#     if args.method == 'gnnsafe':
#         if args.use_prop:
#             name = 'gnnsafe++' if args.use_reg else 'gnnsafe'
#         else:
#             name = 'gnnsafe++ w/o prop' if args.use_reg else 'gnnsafe w/o prop'
#     else:
#         name = args.method
#
#     print(f"Saving results to {filename}")
#     with open(f"{filename}", 'a+') as write_obj:
#         write_obj.write(f"{name} {args.backbone}\n")
#         if args.print_args:
#             write_obj.write(f'{args}\n')
#         if results.shape[0] == 1:  # one run
#             auroc, aupr, fpr = [], [], []
#             for k in range(results.shape[1] // 3):
#                 r = results[:, k * 3]
#                 auroc.append(r.mean())
#                 write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ')
#                 r = results[:, k * 3 + 1]
#                 aupr.append(r.mean())
#                 write_obj.write(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ')
#                 r = results[:, k * 3 + 2]
#                 fpr.append(r.mean())
#                 write_obj.write(f'OOD Test {k + 1} Final FPR: {r.mean():.2f}\n')
#             if k > 0:  # for multiple OODTe datasets, return the averaged metrics
#                 write_obj.write(f'OOD Test Averaged Final AUROC: {np.mean(auroc):.2f} ')
#                 write_obj.write(f'OOD Test Averaged Final AUPR: {np.mean(aupr):.2f} ')
#                 write_obj.write(f'OOD Test Averaged Final FPR: {np.mean(fpr):.2f}\n')
#             r = results[:, -1]
#             write_obj.write(f'IND Test Score: {r.mean():.2f}\n')
#         else:  # more than one runs, return std
#             for k in range(results.shape[1] // 3):
#                 r = results[:, k * 3]
#                 write_obj.write(f'OOD Test {k + 1} Final AUROC: {r.mean():.2f} ± {r.std():.2f} ')
#                 r = results[:, k * 3 + 1]
#                 write_obj.write(f'OOD Test {k + 1} Final AUPR: {r.mean():.2f} ± {r.std():.2f} ')
#                 r = results[:, k * 3 + 2]
#                 write_obj.write(f'OOD Test {k + 1} Final FPR: {r.mean():.2f} ± {r.std():.2f}\n')
#             r = results[:, -1]
#             write_obj.write(f'IND Test Score: {r.mean():.2f} ± {r.std():.2f}\n')
#         write_obj.write(f'\n')


class SimpleLogger(object):
    """ Adapted from https://github.com/CUAI/CorrectAndSmooth """
    def __init__(self, desc, param_names, num_values=2):
        self.results = defaultdict(dict)
        self.param_names = tuple(param_names)
        self.used_args = list()
        self.desc = desc
        self.num_values = num_values

    def add_result(self, run, args, values):
        """Takes run=int, args=tuple, value=tuple(float)"""
        assert(len(args) == len(self.param_names))
        assert(len(values) == self.num_values)
        self.results[run][args] = values
        if args not in self.used_args:
            self.used_args.append(args)

    def get_best(self, top_k=1):
        all_results = []
        for args in self.used_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)[-1]
            results_std = results.std(dim=0)

            all_results.append((args, results_mean))
        results = sorted(all_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [i[0] for i in results]

    def prettyprint(self, x):
        if isinstance(x, float):
            return '%.2f' % x
        return str(x)

    def display(self, args = None):

        disp_args = self.used_args if args is None else args
        if len(disp_args) > 1:
            print(f'{self.desc} {self.param_names}, {len(self.results.keys())} runs')
        for args in disp_args:
            results = [i[args] for i in self.results.values() if args in i]
            results = torch.tensor(results)*100
            results_mean = results.mean(dim=0)
            results_std = results.std(dim=0)
            res_str = f'{results_mean[0]:.2f} ± {results_std[0]:.2f}'
            for i in range(1, self.num_values):
                res_str += f' -> {results_mean[i]:.2f} ± {results_std[1]:.2f}'
            print(f'Args {[self.prettyprint(x) for x in args]}: {res_str}')
        if len(disp_args) > 1:
            print()
        return results