import time
import os
import argparse
import torch
import torchvision.transforms as transforms
from gluoncv.torch.utils.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from MobilenetV3 import *
from config import config
from Loss import FocalLoss
from cosine_lr_scheduler import CosineDecayLR

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def train():
    if not os.path.exists(config.MODEL_ROOT):
        os.mkdir(config.MODEL_ROOT)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config.LOG_ROOT)

    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(112, scale=(0.95, 1), ratio=(1, 1))]),
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.RGB_MEAN, std = config.RGB_STD),
    ])

    dataset_train = ImageFolder(config.TRAIN_FILES, train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = config.BATCH_SIZE, pin_memory = True, shuffle=True,
        num_workers = 8, drop_last = True
    )

    NUM_CLASS = train_loader.dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))
    model = mobilenetv3_small()
    loss = FocalLoss()

    model = torch.nn.DataParallel(model, device_ids=DEVICE)
    model = model.cuda(DEVICE)
    model.eval()
    optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': config.LEARNING_RATE}], momentum=config.MOMENTUM)
    DISP_FREQ = len(train_loader) // 100

    NUM_EPOCH_WARM_UP = config.NUM_EPOCH_WARM_UP
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    scheduler = CosineDecayLR(optimizer, T_max=10*len(train_loader), lr_init = config.QUALITY_LR, lr_min = 1e-5, warmup = NUM_BATCH_WARM_UP)

    batch = 0
    step = 0

    for epoch in range(config.NUM_EPOCH):
        model.train()
        _losses = AverageMeter()
        top1 = AverageMeter()
        scaler = torch.cuda.amp.GradScaler()
        for inputs, labels in tqdm(iter(train_loader)):
            inputs = inputs.cuda(DEVICE)
            labels = labels.cuda(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _loss = loss(outputs, labels)
            prec1 = accuracy(outputs.data, labels, topk=(1,))
            _losses.update(_loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            loss = _loss
            optimizer.zero_grad()
            # loss.backward()
            # OPTIMIZER.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {arcface_loss.val:.4f}({arcface_loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch + 1, config.NUM_EPOCH, batch + 1, len(train_loader) * config.NUM_EPOCH,
                    arcface_loss=_losses, top1=top1))
                print("=" * 60)

            batch += 1  # batch index
            scheduler.step(batch)
            if batch % 1000 == 0:
                print(optimizer)
        # training statistics per epoch (buffer for visualization)
        epoch_loss = _losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch + 1, config.NUM_EPOCH, loss=_losses, top1=top1))
        print("=" * 60)



