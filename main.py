import argparse
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from dataset import MyDataset
from config import parser

#global args
args = parser.parse_args()
best_prec1 = 0

def main():
    global args, best_prec1

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        for param in model.parameters():
            param.requires_grad = True #False
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if 'resnet' in args.arch:
        if args.pose_path:
            model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 21)
    elif args.arch == 'alexnet':
        if args.pose_path:
            model.features._modules['0'] = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
        num_ftrs = model.classifier[6].in_features
        model.classifier._modules['6'] = nn.Linear(num_ftrs, 21)
    elif 'vgg' in args.arch:
        if args.pose_path:
            model.features._modules['0'] = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        num_ftrs = model.classifier[6].in_features
        model.classifier._modules['6'] = nn.Linear(num_ftrs, 21)

    # define loss function (criterion) and optimizer
    weight = torch.ones(21)
    weight[0] = 0.0003
    print('background weight = '+str(weight[0]))
    criterion = nn.CrossEntropyLoss(weight=weight)#.cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.enable_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.enable_gpu:
        cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.frame_path, 'val')
    trainbin = os.path.join(args.gt_path, 'thumos14_val_gt.bin')
    valdir = os.path.join(args.frame_path, 'test')
    valbin = os.path.join(args.gt_path, 'thumos14_test_gt.bin')
    if args.pose_path:
        trainpose = os.path.join(args.pose_path, 'val.bin')
        valpose = os.path.join(args.pose_path, 'test.bin')
    else:
        trainpose = None
        valpose = None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print('start preparing dataset')
    train_dataset = MyDataset(
        'validation', traindir, trainpose, trainbin,
        transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = MyDataset(
        'test', valdir, valpose, valbin,
        transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,#(train_sampler is None),
        num_workers=args.workers, pin_memory=args.enable_gpu)#, sampler=train_sampler)
    # Host to GPU copies are much faster when they originate from pinned (page-locked) memory.

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=args.enable_gpu)

    if args.evaluate:
        validate(val_loader, model, criterion, -1)
        return

    print('start training')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        print('++++++++++++++++++++++++++++++++++++')
        print(time.ctime())
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.eval_freq == 0 or epoch == args.epochs-1 :
            print('====================================')
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                }, is_best)

    if args.pose_path:
        train_dataset.fpose.close()
        val_dataset.fpose.close()


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for i, (videoid, input, target) in enumerate(train_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)#.cuda()
        target_var = torch.autograd.Variable(target)#.cuda()
        if args.enable_gpu:
            # once you pin a tensor or storage, you can use asynchronous GPU
            # copies. Just pass an additional async=True argument to a cuda()
            # call. This can be used to overlap data transfers with computation.
            target = target.cuda(async=True)
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var) # [torch.FloatTensor of size 256x21]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var.data)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), loss=losses, top1=top1))

    print(' >>> Train: Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, loss=losses))


def validate(val_loader, model, criterion, epoch):
    if epoch == -1:
        model.eval()
        predict = {}

        for i, (videoid, input, target) in enumerate(val_loader):
            # Volatile is recommended for purely inference mode, when you’re
            # sure you won’t be even calling .backward(). It’s more efficient
            # than any other autograd setting - it will use the absolute minimal
            # amount of memory to evaluate the model. volatile also determines
            # that requires_grad is False.
            input_var = torch.autograd.Variable(input, volatile=True)#.cuda()
            target_var = torch.autograd.Variable(target, volatile=True)#.cuda()
            if args.enable_gpu:
                target = target.cuda(async=True)
                input_var = input_var.cuda()
                target_var = target_var.cuda()

            # compute output
            output = model(input_var)

            # save predicted output
            for idx, vid in enumerate(videoid): # batch_size is 256
                res = np.expand_dims(output.cpu().data.numpy()[idx], axis=0)
                if vid not in predict:
                    predict[vid] = res
                else:
                    predict[vid] = np.concatenate((predict[vid],res),axis=0)

        save_output(predict)
        return -1

    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (videoid, input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)#.cuda()
        target_var = torch.autograd.Variable(target, volatile=True)#.cuda()
        if args.enable_gpu:
            target = target.cuda(async=True)
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var.data)[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), loss=losses, top1=top1))

    print(' >>> Validate: Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'
          .format(top1=top1, loss=losses))

    return top1.avg

def save_output(predict):
#    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in sorted(predict.items())]))
#    df.to_csv('thumos14_'+mode+'_predict.csv', index=False)
    p = np.array(np.concatenate([v for k,v in sorted(predict.items())], axis=0), dtype=np.float)
    lp = np.array([v.shape[0] for k,v in sorted(predict.items())], dtype=np.int32)
    with open('thumos14_test_' +args.arch+'.bin','wb') as f:
        np.array([len(predict)],dtype=np.int32).tofile(f) # num of videos
        lp.tofile(f) # num of frames per video
        p.tofile(f) # N * 21

def save_checkpoint(state, is_best, filename=args.arch+'_checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.arch+'_model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
