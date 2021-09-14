import time
import os
import argparse
import torch
import torchvision.transforms as transforms
from torch.testing._internal.common_quantization import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from MobilenetV3 import *
from config import config
from Loss import FocalLoss
from cosine_lr_scheduler import CosineDecayLR

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val   = val
        self.sum   += val * n
        self.count += n
        self.avg   = self.sum / self.count

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

def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def convert_target_to_target_format(targets):
    glasses_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)
    mask_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)
    hat_target = torch.zeros(len(targets), dtype=torch.long).cuda(0)

    for idx, target in enumerate(targets):
        if target == 0:
            glasses_target[idx] = 1
            mask_target[idx] = 0
            hat_target[idx] = 0
        elif target == 1:
            glasses_target[idx] = 1
            mask_target[idx] = 0
            hat_target[idx] = 1
        elif target == 2:
            glasses_target[idx] = 1
            mask_target[idx] = 1
            hat_target[idx] = 0
        elif target == 3:
            glasses_target[idx] = 0
            mask_target[idx] = 0
            hat_target[idx] = 1
        elif target == 4:
            glasses_target[idx] = 1
            mask_target[idx] = 1
            hat_target[idx] = 0
        elif target == 5:
            glasses_target[idx] = 1
            mask_target[idx] = 1
            hat_target[idx] = 1
        elif target == 6:
            glasses_target[idx] = 0
            mask_target[idx] = 1
            hat_target[idx] = 1
        elif target == 7:
            glasses_target[idx] = 0
            mask_target[idx] = 0
            hat_target[idx] = 0
    return glasses_target, mask_target, hat_target

def train():
    if not os.path.exists(config.MODEL_ROOT):
        os.mkdir(config.MODEL_ROOT)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config.LOG_ROOT)

    train_transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.01),
        transforms.RandomAutocontrast(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.RGB_MEAN, std = config.RGB_STD),
    ])
    dataset_train = ImageFolder(config.TRAIN_FILES, train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=True,
        num_workers=8, drop_last=True
    )

    valid_transform = transforms.Compose([
        transforms.Resize(config.INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.01),
        transforms.RandomAutocontrast(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.RGB_MEAN, std=config.RGB_STD),
    ])
    dataset_valid = ImageFolder(config.VALID_FILES, valid_transform)
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=config.BATCH_SIZE, pin_memory=True, shuffle=True,
        num_workers=8, drop_last=True
    )

    NUM_CLASS = train_loader.dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))
    model = mobilenetv3_small()
    LOSS = FocalLoss()

    model = torch.nn.DataParallel(model, device_ids=config.DEVICE)
    model = model.cuda(DEVICE)

    if config.PRETRAINED_MODEL is not None:
        load_state_dict(model, torch.load(config.PRETRAINED_MODEL, map_location="cpu"))

    model.eval()
    # optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': config.LEARNING_RATE}], momentum=config.MOMENTUM)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)
    DISP_FREQ = len(train_loader) // 10

    NUM_EPOCH_WARM_UP = config.NUM_EPOCH_WARM_UP
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    # scheduler = CosineDecayLR(optimizer, T_max=10*len(train_loader), lr_init = config.LEARNING_RATE, lr_min = 1e-5, warmup = NUM_BATCH_WARM_UP)

    batch = 0
    step = 0

    for epoch in range(config.NUM_EPOCH):
        model.train()
        _losses = AverageMeter()
        glasses_top1 = AverageMeter()
        mask_top1 = AverageMeter()
        hat_top1 = AverageMeter()

        glasses_valid_top1 = AverageMeter()
        mask_valid_top1 = AverageMeter()
        hat_valid_top1 = AverageMeter()

        scaler = torch.cuda.amp.GradScaler()
        for inputs, labels in tqdm(iter(train_loader)):
            inputs = inputs.cuda(DEVICE)
            labels = labels.cuda(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _loss = LOSS(outputs, labels)
                glasses_target, mask_target, hat_target = convert_target_to_target_format(labels)
                glasses_outputs, mask_output, hat_output = outputs
            glasses_prec1 = accuracy(glasses_outputs.data, glasses_target, topk=(1,))[0]
            mask_prec1 = accuracy(mask_output.data, mask_target, topk=(1,))[0]
            hat_prec1 = accuracy(hat_output.data, hat_target, topk=(1,))[0]
            _losses.update(_loss.data.item(), inputs.size(0))
            glasses_top1.update(glasses_prec1.data.item(), inputs.size(0))
            mask_top1.update(mask_prec1.data.item(), inputs.size(0))
            hat_top1.update(hat_prec1.data.item(), inputs.size(0))
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
                      'Training Glasses Prec@1 {glasses_top1.val:.3f} ({glasses_top1.avg:.3f})\t'
                      'Training Mask Prec@1 {mask_top1.val:.3f} ({mask_top1.avg:.3f})\t'
                      'Training Hat Prec@1 {hat_top1.val:.3f} ({hat_top1.avg:.3f})\t'
                    .format(epoch + 1, config.NUM_EPOCH, batch + 1, len(train_loader) * config.NUM_EPOCH,
                    arcface_loss=_losses, glasses_top1=glasses_top1, mask_top1= mask_top1, hat_top1= hat_top1))
                print("=" * 60)

            batch += 1  # batch index
            # scheduler.step(batch)
            if batch % 1000 == 0:
                print(optimizer)
        # training statistics per epoch (buffer for visualization)
        epoch_loss = _losses.avg
        epoch_acc = (glasses_top1.avg + mask_top1.avg + hat_top1.avg)/3
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Glasses Prec@1 {glasses_top1.val:.3f} ({glasses_top1.avg:.3f})\t'
                      'Training Mask Prec@1 {mask_top1.val:.3f} ({mask_top1.avg:.3f})\t'
                      'Training Hat Prec@1 {hat_top1.val:.3f} ({hat_top1.avg:.3f})\t'
            .format(epoch + 1, config.NUM_EPOCH, loss=_losses, glasses_top1=glasses_top1, mask_top1= mask_top1, hat_top1= hat_top1))
        print("=" * 60)

        for inputs, labels in tqdm(iter(valid_loader)):
            inputs = inputs.cuda(DEVICE)
            labels = labels.cuda(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                glasses_target, mask_target, hat_target = convert_target_to_target_format(labels)
                glasses_outputs, mask_output, hat_output = outputs
            glasses_valid1 = accuracy(glasses_outputs.data, glasses_target, topk=(1,))[0]
            mask_valid1 = accuracy(mask_output.data, mask_target, topk=(1,))[0]
            hat_valid1 = accuracy(hat_output.data, hat_target, topk=(1,))[0]
            _losses.update(_loss.data.item(), inputs.size(0))
            glasses_valid_top1.update(glasses_valid1.data.item(), inputs.size(0))
            mask_valid_top1.update(mask_valid1.data.item(), inputs.size(0))
            hat_valid_top1.update(hat_valid1.data.item(), inputs.size(0))

        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Valid Glasses Prec@1 {glasses_top1.val:.3f} ({glasses_top1.avg:.3f})\t'
              'Valid Mask Prec@1 {mask_top1.val:.3f} ({mask_top1.avg:.3f})\t'
              'Valid Hat Prec@1 {hat_top1.val:.3f} ({hat_top1.avg:.3f})\t'
              .format(epoch + 1, config.NUM_EPOCH, glasses_top1=glasses_valid_top1, mask_top1=mask_valid_top1,
                      hat_top1=hat_valid_top1))

        torch.save(model.state_dict(), os.path.join(config.MODEL_ROOT,
                                                      "Classify_Epoch_{}_Batch_{}_{:.3f}_{:.3f}_{:.3f}_Time_{}_checkpoint.pth".format(
                                                          epoch + 1, batch, glasses_valid_top1.avg, mask_valid_top1.avg,
                                                    hat_valid_top1.avg, time.time())))


if __name__ == "__main__":
    train()
