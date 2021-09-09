import torchvision

import libs
import mydataloader.dataset as dataset
import mynetwork.myresnet as resnet
from torch.utils.data import DataLoader
import torch
import numpy as np
import mydataloader.transforms as tr
from torchvision import transforms as T
import albumentations as A

import argparse
from libs.metrics import get_matrix


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root_dir', default=r'E:\QinGang\data-tag\\train_data', type=str)
    parser.add_argument("--valid_root_dir", default=r'E:\QinGang\data\\train_data', type=str)
    parser.add_argument("--test_root_dir", default=r'E:\QinGang\data\\test_data', type=str)
    parser.add_argument("--model_save", default=r'models\efficientnet-b0\data', type=str)
    parser.add_argument("--k_data_num", default=6, type=int)

    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--load_model", default=False, type=bool)
    parser.add_argument("--load_model_fullname", default='', type=str)

    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epoch_nums", default=14, type=int)
    args = parser.parse_args()
    return args

import time
def train_valid(network, args, train_dataloader, valid_dataloader, test_dataloader, optim, celoss, scheduler):
    history = []
    best_acc = 0.0
    best_epoch = 0
    best_f1 = 0.0
    best_test_classes = []
    epoch_nums = args.epoch_nums
    network = network.cuda()

    for epoch in range(epoch_nums):
        if epoch - best_epoch >= 20:
            break

        train_TP = [0] * 4
        train_TN = [0] * 4
        train_TP_and_FP = [0] * 4
        train_TP_and_FN = [0] * 4
        train_precision = [0] * 4
        train_recall = [0] * 4
        train_f1 = [0] * 4
        train_loss = 0

        valid_TP_and_FP = [0] * 4
        valid_TP_and_FN = [0]*4
        valid_TP = [0]*4
        valid_TN = [0]*4
        valid_precision = [0]*4
        valid_recall = [0]*4
        valid_f1 = [0]*4
        valid_loss = 0
        test_classes = []
        train_tmp_prediction = torch.Tensor([]).cuda()
        train_tmp_label = torch.Tensor([]).cuda()
        valid_tmp_prediction = torch.Tensor([]).cuda()
        valid_tmp_label = torch.Tensor([]).cuda()

        gpu_time = 0
        data_time = 0
        data_start_time = time.time()
        network.train()
        for i, (image, label) in enumerate(train_dataloader):   #len为(train_dataloader/16)
            if (image.shape[1] !=3):  #只考虑三通道的数据，，，只考虑图片尺寸H*W<2000的数据
                continue
            if (image.shape[1] !=3) | (image.shape[2] > 2000) | (image.shape[3] > 2000):
                continue
            optim.zero_grad()
            data_end_time = time.time()
            gpu_start_time = time.time()
            image = image.cuda()
            # print(f"resnet_train_valid 34:image.device:{image.device}")  # cpu
            label = label.cuda()
            result = network(image)

            loss = celoss(result, label)   #result(16,4)   label(16)

            train_loss += loss
            ret, prediction = torch.max(result, dim=1)   #(16)
            train_tmp_prediction = torch.cat([train_tmp_prediction, prediction], axis=0)   #prediciton(16)->(32)->(48)->(64)->train_tmp_prediction(1328)
            train_tmp_label = torch.cat([train_tmp_label, label], axis=0)

            loss.backward()
            optim.step()
            gpu_end_time = time.time()
            data_time += data_end_time - data_start_time
            gpu_time += gpu_end_time - gpu_start_time
            data_start_time = time.time()
        with torch.no_grad():  #在验证集上判断
            network.eval()
            for i, (image,label) in enumerate(valid_dataloader):
                image = image.cuda()
                label = label.cuda()
                result = network(image)  #(16,4)
                # loss = libs.criteria.nll_loss(result, label)
                loss = celoss(result, label)
                valid_loss += loss
                ret, prediction = torch.max(result, dim=1)  #tensor(16)
                valid_tmp_prediction = torch.cat([valid_tmp_prediction, prediction],axis=0)  #prediciton(16)->(32)->(48)->(64)->train_tmp_prediction(1328)
                valid_tmp_label = torch.cat([valid_tmp_label, label],axis=0)

            for i,(image, image_filename) in enumerate(test_dataloader):
                image = image.cuda()
                result = network(image)
                ret, prediction = torch.max(result, dim=1)
                test_classes.append(prediction.cpu().item())

        scheduler.step()
        #调整学习率。。see
        train_TP, train_TN, train_TP_and_FP, train_TP_and_FN = get_matrix(train_tmp_prediction, train_tmp_label)  #输入参数(1328) array(4)
        train_precision = train_TP / train_TP_and_FP
        train_recall = train_TP / train_TP_and_FN
        train_f1 = 2 * (train_precision*train_recall) / (train_precision+train_recall)
        train_f1_avg = np.average(train_f1)
        avg_train_loss = train_loss / len(train_dataloader)

        valid_TP, valid_TN, valid_TP_and_FP, valid_TP_and_FN = get_matrix(valid_tmp_prediction, valid_tmp_label)   #array(4)
        # print(valid_TP, valid_TN, valid_TP_and_FP, valid_TP_and_FN)
        valid_precision = valid_TP/valid_TP_and_FP     #array(4)
        valid_recall = valid_TP/valid_TP_and_FN
        valid_f1 = 2 * (valid_precision*valid_recall)/ (valid_precision + valid_recall)
        valid_f1_avg = np.average(valid_f1)
        avg_valid_loss = valid_loss / len(valid_dataloader)

        history.append([avg_train_loss.cpu().item(), avg_valid_loss.cpu().item(), train_f1_avg, valid_f1_avg])

        #保存最好的model
        if best_f1 < valid_f1_avg:
            best_f1 = valid_f1_avg
            best_epoch = epoch+1
            best_test_classes = test_classes
            # print(f"best_epoch:{best_epoch},best_f1 :{best_f1}")
            print(
                "{:.6f}\t{:.6f}\tepoch {:02d}/{:02d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\tmodel_saved".format(
                    data_time, gpu_time, epoch + 1, epoch_nums, avg_train_loss, avg_valid_loss, train_f1_avg, valid_f1_avg))
            torch.save(network, os.path.join(args.model_save, str(best_epoch) +'best.pt'))
        else:
            print(
                "{:.6f}\t{:.6f}\tepoch {:02d}/{:02d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
                    data_time, gpu_time, epoch + 1, epoch_nums, avg_train_loss, avg_valid_loss, train_f1_avg, valid_f1_avg))
            torch.save(network, os.path.join(args.model_save, str(epoch+1) + 'normal.pt'))
        # if best_acc < avg_train_acc:
        #     best_acc = avg_train_acc
        #     best_epoch = epoch + 1

        # print(
        #     "epoch {:03d}/{:03d}, avg_train_loss:{:.4f}, avg_train_acc[0级污染]:{:.4f},avg_train_acc[1级污染]:{:.4f}，avg_train_ac[2级污染]:{:.4f}，avg_train_acc[3级污染]:{:.4f}".format(
        #         epoch + 1, args.epoch_nums, avg_train_loss, avg_train_acc[0], avg_train_acc[1], avg_train_acc[2],
        #         avg_train_acc[3]
        #     ))
        # print(
        #     "epoch {:03d}/{:03d}, avg_train_loss:{:.4f}, f1[0级污染]:{:.4f},f1[1级污染]:{:.4f}，f1[2级污染]:{:.4f}，f1[3级污染]:{:.4f}".format(
        #         epoch + 1, args.epoch_nums, avg_train_loss, f1[0], f1[1], f1[2], f1[3]
        #     ))

        # print("best accuracy for train, best_epoch{:03d}, best_accuracy:{:.4f}".format(
        #     best_epoch, best_acc
        # ))

    return network, history, best_test_classes
import os
import matplotlib.pyplot as plt
from libs.criteria import CELoss
from mynetwork.myefficientnet import get_efficientnet_b5, get_efficientnet_b3, get_efficientnet_b0

if __name__=='__main__':
    args = parse_command()
    print(args)

    if args.load_model:
        network = torch.load("models_third_metric_valid_minibatch8_transformer/100.pt")
    else:
        network = get_efficientnet_b0()
        # network.to('cuda:0')
        network= network.cuda()
        # print(next(network.parameters()).is_cuda)  # False

    transformer = T.Compose([
        # tr.FixedResize(size=(512, 512)),
        # 色彩 亮度    噪声    随机剪切
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])0-
        ])
    # print(network)
    train_dataset = dataset.MyDataSet(args.train_root_dir, data_type='train', transformer=transformer, k_data_num=args.k_data_num)
    valid_dataset = dataset.MyDataSet(args.valid_root_dir, data_type='valid', transformer=transformer, k_data_num=5)
    test_dataset = dataset.MyDataSet(args.test_root_dir, data_type='test')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
    # valid_dataloader = iter(valid_dataloader)
    # print(next(valid_dataloader)[:][1])
    # print(next(valid_dataloader)[:][1])
    # print(next(valid_dataloader)[:][1])
    # print(next(valid_dataloader)[:][1])
    # test_dataloader = iter(test_dataloader)
    # print(next(test_dataloader))


    # optim = torch.optim.SGD(network.parameters(), lr=0.0005, momentum=0.2, weight_decay=1e-2)
    loss = CELoss(label_smooth=0.1, class_num=4)
    optim = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=2, T_mult=2, eta_min=1e-5)
    # optim = torch.optim.Adam(network.parameters())

    # lambda1 = lambda epoch: 1/(args.epoch_nums+1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda1)
    train_model, history, best_test_classes = train_valid(network, args, train_dataloader, valid_dataloader, test_dataloader, optim, loss, scheduler)
    print(best_test_classes)
    torch.save(history, os.path.join(args.model_save, 'model_history.pt'))

    history = np.array(history)

    plt.plot(history[:, 0:2])
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch_num')
    plt.ylabel("loss")
    plt.savefig(os.path.join(args.model_save, 'loss_curve.jpg'))
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['train_f1_avg', 'valid_f1_avg'])
    plt.xlabel('epoch_num')
    plt.ylabel('f1_avg')
    plt.savefig(os.path.join(args.model_save, "f1_avg_curve.jpg"))
    plt.show()

    #利用history的数据画图





