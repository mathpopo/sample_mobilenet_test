from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy', 'accuracy1', 'accuracy2']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        m = torch.nn.Softmax(dim=1)
        output = m(output)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = torch.squeeze(pred)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target)
        new_res = correct.float()

        h = torch.sum(new_res) / torch.tensor([8]).cuda()

        res = []
        res.append(h)
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy1(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        m = torch.nn.Softmax(dim=1)
        output = m(output)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = torch.squeeze(pred)
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print("show result:")
        # print(pred)
        # print("show gt:")
        # print(target)
        correct = pred.eq(target)
        new_res = correct.float()

        h = torch.sum(new_res) / torch.tensor([14]).cuda()

        res = []
        res.append(h)
        # for k in topk:
        #     correct_k = correct[:k].view(-1).float().sum(0)
        #     res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy2(output, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        m = torch.nn.Softmax(dim=1)
        output = m(output)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = torch.squeeze(pred)
        return pred
