import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args

import utils.common as utils
from utils.common import *
import json
import os
import time
import math
import numpy as np
import random
from utils.scheduler import get_policy
from utils.conv_type import *
from utils.common import *
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as tmodels
from torch.utils.data import DataLoader, Subset
import models
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
# from fvcore.nn import flop_count, flop_count_table, FlopCountAnalysis
# from fvcore.nn import parameter_count, parameter_count_table
import copy
from pthflops import count_ops
seed = 42
np.random.seed(seed)
# random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

#"./final_experiments/resnet/exp1_mag_grad/pattern_wise/uniform_"
save_dir = args.save_dir

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# load training data
print('==> Preparing data..')

if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    #CIFAR10/CIFAR100
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize images to fit the ResNet input size
        transforms.RandomCrop(224, padding=4),   # Resize images to fit the ResNet input size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #CIFAR10/CIFAR100
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),   # Resize images to fit the ResNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
elif args.dataset == 'imagenet':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose(
            [   transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]
        )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]
        )


batch_size = 128
if args.dataset == 'CIFAR10':
    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'CIFAR100':
    trainset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
elif args.dataset == 'imagenet':
    trainset = dset.ImageFolder(root="datasets/imagenet/train/", 
                                transform=transform_train)
    testset = dset.ImageFolder(root="datasets/imagenet/val/", 
                               transform=transform_test)

selected_cls = random.sample(range(0, args.num_classes), args.num_pref_classes)
num_of_samples = 256 * args.num_pref_classes

cls_indices = [i for i, label in enumerate(trainset.targets) if label in selected_cls]
indices = random.sample(cls_indices, num_of_samples)
train_subset = Subset(trainset, indices)
train_loader = torch.utils.data.DataLoader(
    train_subset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

cls_indices = [i for i, label in enumerate(testset.targets) if label in selected_cls]
val_subset = Subset(testset, cls_indices)
val_loader = torch.utils.data.DataLoader(
   val_subset, batch_size=args.eval_batch_size, shuffle=False, num_workers=8)

def train(epoch, num_epochs, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    num_iter = len(train_loader)

    print_freq = num_iter // 10
    # print(print_freq)
    if print_freq == 0:
        print_freq = 1

    i = 0 
    running_grads = {}
    running_counts = {}
    for batch_idx, (images, targets) in enumerate(train_loader):
        if args.debug:
            if i > 5:
                break
            i += 1
        images = images.cuda()
        targets = targets.cuda()
        data_time.update(time.time() - end)

        adjust_learning_rate(optimizer, epoch, batch_idx, num_iter, num_epochs)

        # compute output
        logits = model(images)
        loss = loss_func(logits, targets)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Add up the gradients for each weight tensor
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name and 'Aux' not in name:
                if name not in running_grads:
                    running_grads[name] = torch.zeros_like(param.grad)
                    running_counts[name] = 0
                running_grads[name] += param.grad.abs()
                running_counts[name] += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.step()
        if batch_idx % print_freq == 0 and batch_idx != 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iter, loss=losses,
                    top1=top1, top5=top5))
    avg_grad = {}
    # Calculate the average gradient for each weight tensor
    for name, param in model.named_parameters():
        if param.requires_grad and 'bias' not in name:
            if name in running_grads:
                avg_grad[name] = running_grads[name] / running_counts[name]
                    # print("Average gradient of %s: %.10f" % (name, avg_grad[name].mean()))

    return losses.avg, top1.avg, top5.avg, avg_grad

def validate(val_loader, model, criterion, args):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        for batch_idx, (images, targets) in enumerate(val_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()
            logits = model(images)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def apply_NM_pruning(param):
    N = args.N
    M = args.M
    weight = param.detach()
    length = weight.numel()
    group = int(length/M)

    weight_temp = weight.detach().abs().clone().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M-N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)
    m = w_b
    return m


def get_model(args):
    model = models.__dict__[args.arch]().to(device)

    if args.arch == "resnet18":
        trained_model = tmodels.resnet18(pretrained = True).to(device)
        ckpt = trained_model.state_dict()
        fc_weight = ckpt['fc.weight']
        ckpt['fc.weight'] = fc_weight.view(
            fc_weight.size(0), fc_weight.size(1), 1, 1)
        model.load_state_dict(ckpt, strict = False)
        # model.avgpool = nn.AvgPool2d(2)

        # Reset the weight parameters of the last layer (fc2 in this case)
        last_layer_name = 'fc'
        if args.dataset != 'imagenet':
            fc_layer = BlockL1Conv(
                    args.N, 
                    args.M,
                    512, args.num_classes, kernel_size=1, stride=1, bias=False
                )
            model.fc = fc_layer.cuda()        


        # # getattr(utils.conv_type, args.conv_type)
        # last_layer = model._modules[last_layer_name]

        # # if isinstance(last_layer, nn.Linear):
        # last_layer.reset_parameters()

        # model.fc = get_builder.conv1x1_fc(512, 10)
        # print('==> Apply 2:4 model pruning <==')
        # for name, param in model.named_parameters():
        #     if ('layer' in name and 'conv' in name and 'weight' in name) or 'fc.weight' in name or 'downsample.0' in name:
        #         m = apply_NM_pruning(param)
        #     if (('layer' in name and 'conv' in name) or 'fc' in name or 'downsample.0' in name) and 'NM_mask' in name:
        #         param.data = m
    elif args.arch == "resnet50":
        trained_model = tmodels.resnet50(pretrained = True).to(device)

        ckpt = trained_model.state_dict()
        fc_weight = ckpt['fc.weight']
        ckpt['fc.weight'] = fc_weight.view(
            fc_weight.size(0), fc_weight.size(1), 1, 1)
        model.load_state_dict(ckpt, strict = False)
        # model.avgpool = nn.AvgPool2d(2)
        # for module in model.modules():
        #     if isinstance(module, BlockL1Conv) and module.planes == 1000:
                # Reset the weight parameters of the last layer (fc2 in this case)
        last_layer_name = 'fc'
        if args.dataset != 'imagenet':
                fc_layer = BlockL1Conv(
                        args.N, 
                        args.M,
                        2048, args.num_classes, kernel_size=1, stride=1, bias=False
                    )
                model.fc = fc_layer.cuda()        

    elif args.arch == "inceptionv3":
        model.aux_logits=False

        print('==> Apply 2:4 model pruning <==')
        for name, param in model.named_parameters():
            if ('weight' in name) and 'Conv2d_1a_3x3' not in name:
                m = apply_NM_pruning(param)
            if  'Conv2d_1a_3x3' not in name and 'NM_mask' in name:
                param.data = m

    elif args.arch == "vgg16_bn":
        if args.dataset != 'imagenet':
            classifier_layer = BlockL1Conv(
                    args.N, 
                    args.M,
                    4096, args.num_classes, kernel_size=1, stride=1, bias=False
                )
            model.classifier[6] = classifier_layer.cuda()

    elif args.arch == "mobilenetv2":
        ckpt = torch.load('/home/shivam/NUS/personalization/mobilenetv2.pytorch/pretrained/mobilenetv2-c5e733a8.pth')
        fc_weight = ckpt['classifier.weight']
        ckpt['classifier.weight'] = fc_weight.view(
            fc_weight.size(0), fc_weight.size(1), 1, 1)
        model.load_state_dict(ckpt, strict = False)
        if args.dataset != 'imagenet':
            classifier_layer = BlockL1Conv(
                    args.N, 
                    args.M,
                    1280, args.num_classes, kernel_size=1, stride=1, bias=False
                )
            model.classifier[1] = classifier_layer.cuda() 

        # print(model)   
        # for name, module in model.named_modules():
        #     print(name)  
        # for module in sparse_model.modules():
        #     if isinstance(module, nn.Linear) and module.out_features == 1000:
        #         print(module)
    #     model.classifier
    #     # print('==> Apply 2:4 model pruning <==')
    #     # for name, param in model.named_parameters():
    #     #     if ('weight' in name) and 'features.0' not in name:
    #     #         m = apply_NM_pruning(param)
    #     #     if  'features.0' not in name and 'NM_mask' in name:
    #     #         param.data = m

    print('==> Testing Baseline Model..')
    # print(model)
    validate(val_loader, model, loss_func, args)

    cfg_len = {
        'resnet18': 21,
        'resnet50': 64,
        'vgg16_bn': 32,
        "inception_v3": 70, 
        "mobilenetv2": 50
    }
    pr_cfg = cfg_len[args.arch]
    
    return model, pr_cfg    

def get_block_prune_weight_score(module, name, avg_grad):
    # if hasattr(module, "NM_mask"):
    #     sparseWeight1 = module.get_NM_sparse_weights().clone()
    #     sparseWeight = sparseWeight1
    #     w = sparseWeight.detach().cpu()
    #     w_grad = avg_grad[name + '.weight'].data.clone().detach().cpu()
    #     w_pruning_score = torch.abs(w * (w_grad))              

    if hasattr(module, "block_mask"):
        sparseWeight2 = module.get_block_sparse_weights().clone()
        sparseWeight = sparseWeight2
        w = sparseWeight.detach().cpu()
        w_grad = avg_grad[name + '.weight'].data.clone().detach().cpu()
        w_pruning_score = torch.abs(w * (w_grad))              
    return w_pruning_score

def get_prune_weight_score(module, name, avg_grad):
    if hasattr(module, "NM_mask"):
        sparseWeight1 = module.get_NM_sparse_weights().clone()
        sparseWeight = sparseWeight1
        w = sparseWeight.detach().cpu()
        w_grad = avg_grad[name + '.weight'].data.clone().detach().cpu()
        w_pruning_score = torch.abs(w * (w_grad))              

    if hasattr(module, "block_mask"):
        sparseWeight2 = module.get_block_sparse_weights().clone()
        sparseWeight = sparseWeight1 * sparseWeight2
        w = sparseWeight.detach().cpu()
        w_grad = avg_grad[name + '.weight'].data.clone().detach().cpu()
        w_pruning_score = torch.abs(w * (w_grad))              
    return w_pruning_score
    # else:    
    #     sparseWeight = sparseWeight1
    #     w = sparseWeight.detach().cpu()
    #     w_grad = avg_grad[name + '.weight'].data.clone().detach().cpu()
    #     w_pruning_score = torch.abs(w * (w_grad))              
    #     return w_pruning_score

def get_prune_column_score(weights, block_sz):
        c_out, c_in, k_1, k_2 = weights.shape
        weights = weights.contiguous().view(-1, c_in*k_1*k_2)

        # print(weights.view(-1).size(0), weights.view(-1).size(0) // (block_sz * block_sz), weights.view(-1).size(0) % (block_sz * block_sz))
        if weights.view(-1).size(0) % (block_sz * block_sz) == 0:

            w_list = [torch.stack(torch.split(weights[i], block_sz), dim=0) for i in range(len(weights))]
            weights = torch.stack(w_list, dim=1)

            #reshaped weights into blocks
            weights = weights.view(-1, block_sz, block_sz)     

            #calculating sum of each block
            w_sum = weights.view(-1, block_sz * block_sz)
            w_sum = torch.sum(torch.abs(w_sum), 1) 

            #reshaped columns for each row
            if w_sum.view(-1).size(0) % (c_in*k_1*k_2 // block_sz) == 0:
                w_sum = w_sum.view(-1, c_in*k_1*k_2 // block_sz)
                w_sum_sorted, w_sum_indices = torch.sort(w_sum)
                return w_sum_sorted, w_sum_indices 
            else:
                return None, None
        else:
            return None, None

def create_column_wise_mask(m, c_out, c_in, k_1, k_2, block_sz):

    # print('--> mask creation')
    m = m.view(-1)
    m = torch.unsqueeze(m, 1)
    m = m.repeat((1, block_sz * block_sz))
    m = m.view(-1, block_sz, block_sz)

    # print('--> index calc')
    total_ele_left = c_in * k_1 * k_2
    total_ele_NM = block_sz
    total_left = total_ele_left // total_ele_NM
    total_right = c_out // block_sz
    m_list = list(m)
    n_c = 0

    w_copy = torch.randint(0, 2, (c_out, total_ele_left))
    # print('--> iterating i in total right', total_right)
    # print('--> iterating j in total left', total_left)
    # print('--> iterating k1 in block sz', block_sz)
    # print('--> iterating k2 in block sz', block_sz)
    # print('--> copying')

    for i in range(total_right):
        for j in range(total_left):
            for k1 in range(block_sz):
                for k2 in range(block_sz):
                    w_copy[i*block_sz + k1, j*block_sz + k2] = m_list[n_c][k1, k2]
            n_c += 1
    m = w_copy
    m = m.view(c_out, c_in, k_1, k_2)  
    return m   



def main():

    ratio  = args.pr_target    
    blocks = [16, 32, 64]

    for blk_id in range(3):        
        print("----Block Size ", blocks[blk_id], "----") 
        PATH = args.save_dir + "/block_size_" + str(blocks[blk_id])
        print(PATH)
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        # directory for saving experimental results
        save_run_dir = os.path.join(PATH, "run_" + str(args.run), "target_ratio_" + str(ratio))
        if not os.path.exists(save_run_dir):
            os.makedirs(save_run_dir)
        filename = os.path.join(PATH, "run_" + str(args.run), "target_ratio_" + str(ratio), "results.json")
        if os.path.exists(filename):
            os.remove(filename)
        outfile =  open(filename, "a") 

        best_acc = 0.0
        best_acc_top1 = 0.0

        model, pr_cfg = get_model(args)
        optimizer = get_optimizer(args, model)

        blc_szs = [blocks[blk_id]] * pr_cfg
        per_layer_pre_prune_wdensity = [0] * pr_cfg
        per_layer_post_prune_wdensity = [0] * pr_cfg

        print('-----Block Pruning initiation stage-----')

        print('----Finetune stage-----')
        for epoch in range(0, 1):
            
            train_obj, train_acc_top1,  train_acc, avg_grad = train(epoch, 1, train_loader, model, loss_func, optimizer)
            valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)

            model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
            num_zeros, num_elements, sparsity = measure_global_sparsity(
                model, weight = True,
                bias = False, conv2d_use_mask = True,
                linear_use_mask = False)                
            print(f"Global sparsity = {sparsity * 100:.3f}%")

        igp_id = 0
        num_epochs = args.finetune_epochs * (int(ratio*10) + 1)
        for target_ratio in range(0, int(ratio*10) + 1, 1):
        # for target_ratio in range(0, int(ratio*10) + 1, 1):
            
            print('-------------------------------')
            print('Target pruning ratio:', target_ratio*5 + sparsity * 100)

            num_zeros, num_elements, sparsity = measure_global_sparsity(
                model, weight = True,
                bias = False, conv2d_use_mask = True,
                linear_use_mask = False)
            
            print(f"Sparsity before block pruning = {sparsity * 100:.3f}%")

            to_concat_weight = []
            total_param = 0
            layer_scores = dict()
            layer_not_block_pruned = ['classifier.6', 'classifier.0', 'features.0', 'fc', 'features.0.0', 'features.2.conv.6', 'features.3.conv.0', 'features.3.conv.6', 'features.4.conv.0', 'classifier.1']

            for name, module in model.named_modules():
                if name not in layer_not_block_pruned:
                # if 'Conv2d_1a_3x3' not in name and 'fc' not in name and 'Aux' not in name and 'features.0' not in name and 'classifier.6' not in name:
                    # print(name)
                # if  ('layer' in name and 'conv' in name) or 'downsample.0' in name:                   
                    if hasattr(module, "NM_mask") and hasattr(module, "block_mask"):
                        # print('calculating block masks', name)
                        w = get_prune_weight_score(module, name, avg_grad)

                        w_sum_sorted, w_sum_indices =  get_prune_column_score(w, blocks[blk_id])
                        layer_scores[name] = [w_sum_sorted, w_sum_indices]
                        if layer_scores[name][0] != None:
                            
                            #calculate column-wise scores
                            w_col_scores = torch.sum(w_sum_sorted, 0)
                            w_col_indices = w_sum_indices
                        
                            #score for each column for layer l
                            to_concat_weight.append(w_col_scores)
                            #total number of blocks in layer l
                            total_param += w_col_scores.size(0) 

            #list of sum of blocks of all layers                        
            all_w = torch.cat(to_concat_weight)

            num_params = total_param 
            thresh = 0 

            nz = int((1 - (target_ratio) * 0.1) * num_params)
            print(nz, target_ratio)
            top_values, _ = torch.topk(all_w, nz)
            thresh = top_values[-1]
            print('threshold', thresh)

            cnt = 0
            for name, module in model.named_modules():
                # if  ('layer' in name and 'conv' in name) or 'downsample.0' in name:
                # if 'Conv2d_1a_3x3' not in name and 'fc' not in name and 'Aux' not in name and 'features.0' not in name and 'classifier3.0' not in name:
                if name not in layer_not_block_pruned:
                    if hasattr(module, "NM_mask") and hasattr(module, "block_mask"):
                        # print('apply block mask', name)
                        w = get_prune_weight_score(module, name, avg_grad)
                        c_out, c_in, k_1, k_2 = w.shape
                        if layer_scores[name][0] != None:
                            # print('calculating column scores')
                            [w_sum_sorted, w_sum_indices] = layer_scores[name]

                            #calculate column-wise scores
                            w_col_scores = torch.sum(w_sum_sorted, 0)
                            w_col_indices = w_sum_indices

                            # print('creating column mask')
                            #create a column-wise mask
                            m_col = torch.zeros_like(w_col_indices)
                            m_col = (w_col_scores > thresh).type(torch.float)
                            m = torch.zeros_like(w_sum_sorted)

                            # print('iterating column mask')
                            for i in range(w_col_indices.shape[0]):
                                jc = 0
                                for j in range(w_col_indices.shape[1]):
                                    m[i][w_col_indices[i][j]] = m_col[jc]
                                    jc = jc + 1

                            # print('applying column mask')
                            pruning_mask = create_column_wise_mask(m, c_out, c_in, k_1, k_2, blocks[blk_id])
                            module.block_mask = nn.Parameter(pruning_mask * module.block_mask.data.cpu(), requires_grad=False)
                            # per_layer_post_prune_wdensity[cnt] = torch.count_nonzero(module.block_mask) / module.block_mask.view(-1).shape[0]
                            cnt = cnt + 1

            model = model.to(device)

            print('==> Apply N:M model pruning <==')
            for name, module in model.named_modules():
                if hasattr(module, "block_mask"):
                    if 'features.0' not in name:
                    # if ('layer' in name and 'conv' in name and 'weight' in name) or 'fc.weight' in name or 'downsample.0' in name:
                        print(name)
                        w = get_block_prune_weight_score(module, name, avg_grad)
                        NM_mask = apply_NM_pruning(w)
                        module.NM_mask = nn.Parameter(NM_mask * module.NM_mask.data.cpu(), requires_grad=False)

                # if ('NM_mask' in name) and 'features.0' not in name and 'Conv2d_1a_3x3' not in name:
                # if (('layer' in name and 'conv' in name) or 'fc' in name or 'downsample.0' in name) and 'NM_mask' in name:
                    # param.data = m
            model = model.to(device)

            #-----------------------------------------------------------------------------------------
            # #create a copy of the sparse model
            if args.arch == 'resnet18':
                sparse_model = (tmodels.resnet18(pretrained = True).to(device))
                for module in sparse_model.modules():
                    if isinstance(module, nn.Linear) and module.out_features == 1000:
                        fc_layer = nn.Linear(
                                module.in_features, args.num_classes, bias=False
                            )
                        sparse_model.fc = fc_layer.cuda()   
            elif args.arch == 'vgg16_bn':
                sparse_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True).cuda()
                for module in sparse_model.modules():
                    if isinstance(module, nn.Linear) and module.out_features == 1000:
                        classifier = nn.Linear(
                                module.in_features, args.num_classes, bias=False
                            )
                        sparse_model.classifier[6] = classifier.cuda()   
            elif args.arch == 'resnet50':
                sparse_model = (tmodels.resnet50(pretrained = True).to(device))
                for module in sparse_model.modules():
                    if isinstance(module, nn.Linear) and module.out_features == 1000:
                        fc_layer = nn.Linear(
                                module.in_features, args.num_classes, bias=False
                            )
                        sparse_model.fc = fc_layer.cuda()   
            elif args.arch == 'mobilenetv2':
                # sparse_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
                # print(sparse_model)
                sparse_model = models.__dict__["mobilenetv2_og"]().to(device)
                for module in sparse_model.modules():
                    if isinstance(module, nn.Linear) and module.out_features == 1000:
                        fc_layer = nn.Linear(
                                module.in_features, args.num_classes, bias=False
                            )
                        sparse_model.classifier[1] = fc_layer.cuda()   



                # sparse_model = (tmodels.resnet18(pretrained = True).to(device))
                # ckpt = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')

            # Create random input tensor
            batch_size, channels, height, width = 1, 3, 224, 224  # Adjust as needed
            input_tensor = torch.randn(batch_size, channels, height, width).cuda()

            # print("Dense Model")
            dense_FLOPs, dense_FLOP_dict = count_ops(sparse_model, input_tensor)
            # Get a list of sparse modules in the original model
            original_modules = [(name, module) for name, module in model.named_modules() if hasattr(module, 'NM_mask')]

            # Create a dictionary to map original module names to new module names
            module_name_mapping = {}

            # Iterate through sparse modules and find corresponding modules in the new model
            for new_name, new_module in sparse_model.named_modules():
                for og_name, og_module in original_modules:
                    if new_name == og_name:
                        module_name_mapping[og_name] = new_module
                        if hasattr(og_module, 'NM_mask') and hasattr(og_module, 'block_mask'):                            
                            if args.arch == 'resnet18' and new_name == 'fc':
                                new_module.weight = nn.Parameter((og_module.NM_mask * og_module.block_mask * og_module.weight.clone()).view(
                                                    new_module.weight.size(0), new_module.weight.size(1)))
                            elif args.arch == 'resnet50' and new_name == 'fc':
                                new_module.weight = nn.Parameter((og_module.NM_mask * og_module.block_mask * og_module.weight.clone()).view(
                                                    new_module.weight.size(0), new_module.weight.size(1)))
                            elif args.arch == 'vgg16_bn' and (new_name == 'classifier.0' or new_name == 'classifier.3' or new_name == 'classifier.6'):
                                new_module.weight = nn.Parameter((og_module.NM_mask * og_module.block_mask * og_module.weight.clone()).view(
                                                    new_module.weight.size(0), new_module.weight.size(1)))
                            elif args.arch == 'mobilenetv2' and (new_name == 'classifier.1'):
                                new_module.weight = nn.Parameter((og_module.NM_mask * og_module.block_mask * og_module.weight.clone()).view(
                                                    new_module.weight.size(0), new_module.weight.size(1)))
                            else:
                                new_module.weight = nn.Parameter(og_module.NM_mask * og_module.block_mask * og_module.weight.clone())
                        elif hasattr(og_module, 'NM_mask'):
                            new_module.weight = nn.Parameter(og_module.NM_mask * og_module.weight.clone())
                        break

            #-----------------------------------------------------------------------------------------
            # print("Sparse Model FLOPs")
            sparse_FLOPs, sparse_FLOPs_dict = (count_ops(sparse_model, input_tensor))
            flop_ratio = sparse_FLOPs / dense_FLOPs
            print(f"Sparse Model FLOPs ratio = {flop_ratio}")

            num_zeros, num_elements, sparsity = measure_global_sparsity(
                model, weight = True,
                bias = False, conv2d_use_mask = True,
                linear_use_mask = False)
            
            print(f"Global sparsity = {sparsity * 100:.3f}%")

            best_acc = 0.0
            best_acc_top1 = 0.0

            print(f"Accuracy before finetuning")
            valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)

            print("---------Finetuning--------")
            for epoch in range(0, args.finetune_epochs):

                iter_epoch = args.finetune_epochs * igp_id + epoch
                train_obj, train_acc_top1,  train_acc, avg_grad = train(iter_epoch, num_epochs, train_loader, model, loss_func, optimizer)
                valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)

                is_best = best_acc_top1 < test_acc_top1
                best_acc_top1 = max(best_acc_top1, test_acc_top1)
                best_acc = max(best_acc, test_acc)

                model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()
                num_zeros, num_elements, sparsity = measure_global_sparsity(
                    model, weight = True,
                    bias = False, conv2d_use_mask = True,
                    linear_use_mask = False)
                
                print(f"Global sparsity = {sparsity * 100:.3f}%")

                # Data to be written
                dictionary = {
                    "sparsity": float(f"{sparsity * 100:.3f}"),
                    "finetune epoch": epoch + 1,
                    "overall epoch": iter_epoch + 1,
                    "top1": float(f"{test_acc_top1:.3f}"),
                    "top5": float(f"{test_acc:.3f}"), 
                    "flops_ratio":  float(f"{flop_ratio:.2f}")
                }

                json.dump(dictionary, outfile)
                outfile.write('\n')

                # args.model_dir = os.path.join(save_run_dir, "sparsity_" + str(ratio*10))
                # checkpoint = utils.checkpoint(args)
                # if is_best:                                    
                #     state = {
                #         'state_dict': model_state_dict,
                #         'best_acc': best_acc_top1,
                #         'optimizer': optimizer.state_dict(),
                #         'epoch': epoch + 1,
                #         # 'overall epoch':
                #         'sparsity': sparsity * 100,
                #     }
                    # checkpoint.save_model(state, epoch + 1, is_best)
                
                print('==> Testing Final Model..')
                validate(val_loader, model, loss_func, args)

                num_zeros, num_elements, sparsity = measure_global_sparsity(
                    model, weight = True,
                    bias = False, conv2d_use_mask = True,
                    linear_use_mask = False)
                    
                print(f"Global sparsity = {sparsity * 100:.3f}%")                    
            igp_id = igp_id + 1

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_parameters():
            if "mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity

def measure_global_sparsity(
    model, weight = True,
    bias = False, conv2d_use_mask = False,
    linear_use_mask = False):

    num_zeros = 0
    num_elements = 0

    for name, module in model.named_modules():
        module_num_zeros = 0
        module_num_elements = 0
        # if  ('layer' in name and 'conv' in name)  or 'fc' in name or 'downsample.' in name:
        if hasattr(module, "NM_mask"):
            sparseWeight1 = module.get_NM_sparse_weights()
        if hasattr(module, "block_mask"):
            sparseWeight2 = module.get_block_sparse_weights()
            sparseWeight = sparseWeight1 * sparseWeight2

            module_num_zeros = torch.sum(sparseWeight == 0).item()
            module_num_elements = module.weight.data.nelement()

        num_zeros += module_num_zeros
        num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_acc"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")

def adjust_learning_rate(optimizer, epoch, step, len_epoch, num_epochs):
    #Warmup
    if args.lr_policy == 'step':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr * (0.1 ** factor)
    elif args.lr_policy == 'cos':  # cos with warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (num_epochs - 5)))
    elif args.lr_policy == 'exp':
        step = 20
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_policy == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    
    if epoch < 5:
            #factor = epoch // 30
            #lr = args.lr * (0.1 ** factor)
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(args, model):
    if args.optimizer == "sgd":

        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

if __name__ == '__main__':
    main()
