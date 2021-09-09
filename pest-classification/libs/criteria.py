import torch.nn as nn
import torch.nn.functional as F

import torch


def cross_entropy(image, label):
    criteria = nn.CrossEntropyLoss(ignore_index=255)
    loss = criteria(torch.unsqueeze(image, 0), label ) # (4)  image(4,10,256,256) label(4,10,256,256)
    return loss
#计算loss时，result的维数比label的高一维。
def nll_loss(result, label):
    loss_func = nn.NLLLoss()
    loss = loss_func(result, label)
    return loss


class CELoss(nn.Module):
    ''' Cross Entropy Loss with label smoothing '''

    def __init__(self, label_smooth=None, class_num=137):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        '''
        Args:
            pred: prediction of model output    [N, M]
            target: ground truth of sampler [N]
        '''
        eps = 1e-12

        if self.label_smooth is not None:
            logprobs = F.log_softmax(pred, dim=1)  # softmax + log    (16,3)
            target = F.one_hot(target, self.class_num)  # 转换成one-hot
            target = torch.clamp(target.float(), min=self.label_smooth / (self.class_num - 1),
                               max=1.0 - self.label_smooth)
            loss = -1 * torch.sum(target * logprobs, 1)  #(16,3)

        else:

            loss = -1. * torch.log(torch.exp(pred + eps).gather(1, target.unsqueeze(0)))

        return loss.mean()
from torch.autograd import Variable
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)

        #标签平滑，针对ids处理，
        #(4, 4)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]   #([1,1]) 调控不均衡样本

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p  #(4,1)
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


if __name__=='__main__':
    #测试focalloss
    # loss1 = nn.CrossEntropyLoss()
    # loss2 = CELoss(label_smooth=0.05, class_num=3)
    loss3 = FocalLoss(class_num=3)
    x = torch.tensor([[1, 8, 1], [1, 1, 8]], dtype=torch.float)
    y = torch.tensor([1, 2])

    print(loss3(x, y))
    # print(loss1(x, y), loss2(x, y))


