# -*- coding: utf-8 -*-

from __future__ import print_function, division
from shutil import copyfile
import yaml
from random_erasing import RandomErasing
from model import ft_net, ft_net_dense, PCB
import os
import time
import matplotlib.pyplot as plt
from datafolder.folder import Train_Dataset

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
# from PIL import Image

version = torch.__version__
# fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError:  # will be 3.x series
    print(
        'This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='output model name')
parser.add_argument('--data_dir', default='../../../dataset',
                    type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true',
                    help='use all training data')
parser.add_argument('--color_jitter', action='store_true',
                    help='use color jitter in training')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float,
                    help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--fp16', action='store_true',
                    help='use float16 instead of float32, which will save about 50% memory')
opt = parser.parse_args()

fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

# two dataset dict
dataset_dict = {
    'market': 'Market-1501',
    'duke': 'DukeMTMC-reID',
}
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.PCB:
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform_val_list = [
        transforms.Resize(size=(384, 192), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + \
        [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'

image_datasets = {}
# image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
#                                           data_transforms['train'])
# image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
#                                           data_transforms['val'])

# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=True, num_workers=8, pin_memory=True)  # 8 workers may work faster
#               for x in ['train', 'val']}

image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict['market'],
                                        train_val='train')
image_datasets['val'] = Train_Dataset(data_dir, dataset_name=dataset_dict['market'],
                                      train_val='query')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
id_class_number = image_datasets['train'].num_id()
attr_class_number = image_datasets['train'].num_label()

use_gpu = torch.cuda.is_available()

since = time.time()
# inputs, classes = next(iter(dataloaders['train']))
print(time.time() - since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, APR_factor=6):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_APR_loss = 0.0
            # Rank@1:0.897862 Rank@5:0.964371 Rank@10:0.978325 mAP:0.741666
            running_loss = 0.0
            running_corrects = 0.0
            running_label_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, id_labels, attr_labels, ids, cams, names = data
                attr_labels = attr_labels.t()
                # images, indices, labels, ids, cams, names = data
                # print(labels)
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    # detach() returns a new Tensor, deta
                    inputs = Variable(inputs.cuda().detach())
                    id_labels = Variable(id_labels.cuda().detach())
                    attr_labels = Variable(attr_labels.cuda().detach())
                else:
                    inputs, id_labels = Variable(inputs), Variable(id_labels)
                # if we use low precision, input also need to be fp16
                # if fp16:
                #    inputs = inputs.half()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)

                label_output = outputs[:-1]  # 30 * 32 * 2
                id_output = outputs[-1]
                if not opt.PCB:
                    _, preds = torch.max(id_output.data, 1)
                    id_loss = criterion(id_output, id_labels)

                    attr_loss = criterion(label_output[0], attr_labels[0])
                    _, pred_attr = torch.max(label_output[0], 1)
                    running_label_corrects += float(
                        torch.sum(pred_attr == attr_labels[0].data).item()) / attr_class_number
                    for i in range(1, attr_class_number):
                        attr_loss += criterion(label_output[i], attr_labels[i])
                        _, pred_attr = torch.max(label_output[i], 1)
                        running_label_corrects += float(
                            torch.sum(pred_attr == attr_labels[i].data).item()) / attr_class_number
                    APR_loss = id_loss * APR_factor + attr_loss / attr_class_number
                    # print(labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) + sm(part[2]) + \
                        sm(part[3]) + sm(part[4]) + sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    id_loss = criterion(part[0], id_labels)
                    for i in range(num_part - 1):
                        id_loss += criterion(part[i + 1], id_labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(id_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        APR_loss.backward()
                        # id_loss.backward()
                    optimizer.step()

                # statistics
                # for the new version like 0.4.0, 0.5.0 and 1.0.0
                if int(version[0]) > 0 or int(version[2]) > 3:
                    running_loss += id_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += id_loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == id_labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, 'last')
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join('./model', name, 'train.jpg'))


######################################################################
# Save model
# ---------------------------


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if opt.use_dense:
    model = ft_net_dense((id_class_number), opt.droprate)
else:
    model = ft_net((id_class_number), attr_class_number,
                   opt.droprate, opt.stride)

if opt.PCB:
    model = PCB((id_class_number))

opt.nclasses = (id_class_number)

print(model)

if not opt.PCB:
    ignored_params = list(map(id, model.model.fc.parameters()))
    for i in range(attr_class_number + 1):
        list(map(id, model.__getattr__('class_' + str(i)).parameters()))

    # ignored_params = list(map(id, model.model.fc.parameters())) + \
    #                  list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.model.fc.parameters(), 'lr': opt.lr},
        # {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], lr=opt.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (list(map(id, model.classifier0.parameters()))
                       + list(map(id, model.classifier1.parameters()))
                       + list(map(id, model.classifier2.parameters()))
                       + list(map(id, model.classifier3.parameters()))
                       + list(map(id, model.classifier4.parameters()))
                       + list(map(id, model.classifier5.parameters()))
                       # +list(map(id, model.classifier6.parameters() ))
                       # +list(map(id, model.classifier7.parameters() ))
                       )
    base_params = filter(lambda p: id(
        p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.model.fc.parameters(), 'lr': opt.lr},
        {'params': model.classifier0.parameters(), 'lr': opt.lr},
        {'params': model.classifier1.parameters(), 'lr': opt.lr},
        {'params': model.classifier2.parameters(), 'lr': opt.lr},
        {'params': model.classifier3.parameters(), 'lr': opt.lr},
        {'params': model.classifier4.parameters(), 'lr': opt.lr},
        {'params': model.classifier5.parameters(), 'lr': opt.lr},
        # {'params': model.classifier6.parameters(), 'lr': 0.01},
        # {'params': model.classifier7.parameters(), 'lr': 0.01}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(
    optimizer_ft, [40, 60, 80], gamma=0.1, last_epoch=-1)
######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run
copyfile('./train.py', dir_name + '/train.py')
copyfile('./model.py', dir_name + '/model.py')

# save opts
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()
if fp16:
    # model = network_to_half(model)
    # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

criterion = nn.CrossEntropyLoss()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=100)
