import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
# topk=(1,5)
# maxk = max(topk)
# print("max",maxk)
# output = torch.tensor([[1,2,3,0,8],[1,0,9,0,1]])
# target = torch.tensor([[4],[0]])
# batch_size = target.size(0)
# print('batch_size',batch_size)
#
# _,pred = output.topk(maxk,1,True,True)
# print('pred',pred)
# print('_',_)
# pred = pred.t()
# print('pred.t',pred)
# target = target.view(1,-1)
# print('targer',target)
# target = target.view(1,-1).expand_as(pred)
# print('targer',target)
# correct = pred.eq(target)
# print('correct',correct)
# # accuracy(output,target)
# k = 5
#
# correct_k = correct[:k].contiguous().view(-1).float().sum(0,keepdim=True)
# print('correct_k',correct_k)
# res = []
# res.append(correct_k.mul_(100 / batch_size))
# print(res)


