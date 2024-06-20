import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import numpy as np
import shutil
import inspect
from dataset import VideoDataset
from transforms import *
from opts import parser
#from models import TSN
from sklearn.metrics import confusion_matrix
import CosineAnnealingLR
import os
from temporal_models import TemporalModel
from tensorboardX import SummaryWriter
import datasets_video
best_prec1 = 0
def init_seed():
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    global args, best_prec1
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    #init_seed()
    
    if args.dataset == 'something-v1':
        num_class = 174
    elif args.dataset == 'diving48':
        num_class = 48
    elif args.dataset == 'ucf101':
        num_class = 101
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    model_dir = os.path.join('experiments', args.dataset, args.arch, args.consensus_type + '-' + args.modality,
                             str(args.run_iter))
    args.store_name = '_'.join([args.dataset, args.arch, args.consensus_type, 'segment%d' % args.num_segments])
    print('storing name: ' + args.store_name)
    if not args.resume_rgb:
        if os.path.exists(model_dir):
            answer = input('delete it? y/n:')
            if str(answer)=='y':
                print('Dir {} exists!!!  it will be removed'.format(model_dir))
                shutil.rmtree(model_dir)
                os.makedirs(model_dir)
                os.makedirs(os.path.join(model_dir, args.root_log))
            else:
                print('Dir not removed: ', arg.work_dir)
        else:
            os.makedirs(model_dir)
            os.makedirs(os.path.join(model_dir, args.root_log))
    log_training = open(os.path.join(model_dir, args.root_log, '%s.csv' % args.store_name), 'a')
    args.train_list, args.val_list, args.root_path, args.rgb_prefix = datasets_video.return_dataset(args.dataset)
    if 'something' in args.dataset:
        # label transformation for left/right categories
        target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
        output ='Target transformation is enabled....'
        print(output)
        log_training.write(output + '\n')
        log_training.flush()
    else:
        target_transforms = None
        
    if args.resume_rgb :
        if args.modality=='RGB':
            if 'gst' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'GST', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'stm' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'STM', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'tmp' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'TMP', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'tsm' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'TSM', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'ori' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'ORI', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            else:
                model = TemporalModel(num_class, args.num_segments, model = 'ORI', backbone=args.arch+'_ori', alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            if os.path.isfile(args.resume_rgb):
                output = "=> loading checkpoint '{}'".format(args.resume_rgb)
                print(output)
                log_training.write(output + '\n')
                log_training.flush()
                checkpoint = torch.load(args.resume_rgb)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                original_checkpoint=checkpoint['state_dict']
                output = "(epoch {} ) best_prec1 : {} ".format(checkpoint['epoch'] , best_prec1)
                print(output)
                log_training.write(output + '\n')
                log_training.flush()
                original_checkpoint = {k[7:] : v for k, v in original_checkpoint.items() }
                #model_dict =  i3d_model.state_dict()
                #model_dict.update(pretrained_dict)
                model.load_state_dict(original_checkpoint)
                print(("=> loaded checkpoint '{}' (epoch {} ) best_prec1 : {} "
                    .format(args.resume_rgb, checkpoint['epoch'] , best_prec1)))
            else:
                raise ValueError("=> no checkpoint found at '{}'".format(args.resume_rgb))
    else:
        if args.modality=='RGB':
            if 'gst' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'GST', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'stm' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'STM', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'tmp' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'TMP', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'tsm' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'TSM', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            elif 'ori' in args.arch:
                model = TemporalModel(num_class, args.num_segments, model = 'ORI', backbone=args.arch, alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
            else:
                model = TemporalModel(num_class, args.num_segments, model = 'ORI', backbone=args.arch+'_ori', alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout,resi=args.resi)
    
    cudnn.benchmark = True
    writer = SummaryWriter(model_dir)
    # Data loading code
    shutil.copy2(__file__, model_dir)
    import string
    shutil.copy2(args.arch.strip(string.digits)+'.py', model_dir)
    
    total = get_n_params(model)
    output = 'The number of parameters: {}'.format(total)
    print(output)
    log_training.write(output + '\n')
    log_training.flush()
    
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = get_optim_policies(model)
    train_augmentation = get_augmentation(mode='train', target_transforms = target_transforms)
    val_trans = get_augmentation(mode='val')
    normalize = GroupNormalize(input_mean, input_std)
    #print(args.gpus)
    
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    if args.dataset=='diving48':
        args.root_path=args.root_path+'/train'
    train_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.train_list, num_segments=args.num_segments,
                   new_length=args.new_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])
                   ,dataset=args.dataset
                   ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.dataset=='diving48':
        args.root_path=args.root_path[:-6]+'/test'
    val_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list, num_segments=args.num_segments,
                   new_length=args.new_length,
                   modality=args.modality,
                   image_tmpl=args.rgb_prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                       ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                       normalize,
                   ])
                   ,dataset=args.dataset
                   ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")
    
    for group in policies:
        output = 'group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])
        print(output)
        log_training.write(output + '\n')
        log_training.flush()
    
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    if args.evaluate:
        log_test = open('test_not.csv', 'w')
        validate(val_loader,model, criterion,log_test)
        os.remove(log_test)
        return
        
    if args.lr_scheduler=='cos_warmup':
        lr_scheduler_clr = CosineAnnealingLR.WarmupCosineLR(optimizer=optimizer, milestones=[args.warmup, args.epochs], warmup_iters=args.warmup, min_ratio=1e-7)
    elif args.lr_scheduler=='lr_step_warmup':
        lr_scheduler_clr = CosineAnnealingLR.WarmupStepLR(optimizer=optimizer, milestones=[args.warmup]+[args.epochs-30, args.epochs-10, args.epochs], warmup_iters=args.warmup)
    elif args.lr_scheduler=='lr_step':
        lr_scheduler_clr=torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, 0.1)
    if args.resume_rgb:
        for epoch in range(0, args.start_epoch):
            optimizer.step()
            lr_scheduler_clr.step()
    
    for epoch in range(args.start_epoch, args.epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
        train(train_loader, model, criterion, optimizer, epoch, log_training, writer=writer)
        lr_scheduler_clr.step()
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1, cls_acc = validate(val_loader, model, criterion,  log_training, writer=writer, epoch=epoch)
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'lr': optimizer.param_groups[-1]['lr'],
                }, is_best, model_dir)
                np.save(model_dir+'/test_score',cls_acc)
            output = 'best_prec1: {}'.format(best_prec1)
            print(output)
            log_training.write(output + '\n')
            log_training.flush()
        if (epoch + 1) % args.save_interval==0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'lr': optimizer.param_groups[-1]['lr'],
            }, False, model_dir, epoch = epoch + 1)
        #else:
        #    save_checkpoint({
        #        'epoch': epoch + 1,
        #        'arch': args.arch,
        #        'state_dict': model.state_dict(),
        #        'best_prec1': best_prec1,
        #        'lr': optimizer.param_groups[-1]['lr'],
        #    }, False, model_dir)

def get_augmentation(mode='train', target_transforms=None):
    if mode == 'train':
        if args.modality == 'RGB' :
                return torchvision.transforms.Compose(
                [    GroupMultiScaleCrop(224, [1, .875, .75, .66]),
                     GroupRandomHorizontalFlip(target_transforms)])
    elif mode == 'val':
        return None

def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #model.module.partialBN(True)
    # switch to train mode
    model.train()
    loss_summ = 0
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        if len(output.shape)<2:
            output = torch.unsqueeze(output,0)
        loss = criterion(output, target_var)/ args.iter_size
        loss_summ += loss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss_summ.data, input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))


        # compute gradient and do SGD step

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            #if total_norm > args.clip_gradient:
            #    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning
            optimizer.step()
            optimizer.zero_grad()
            loss_summ = 0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output=('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,data_time=data_time, 
                  loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            writer.add_scalar('train/batch_loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_top1Accuracy', top1.avg, epoch * len(train_loader) + i)
            log.write(output + '\n')
            log.flush()
    output = 'Training costs {} mins {} seconds'.format(int(batch_time.sum)//60, int(batch_time.sum)%60)
    print(output)
    log.write(output + '\n')
    log.flush()
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
    return top1.avg


def validate(val_loader, model, criterion, log=None, epoch=0, writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    video_labels=[]
    video_pred=[]
    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            video_labels.extend(target_var.cpu().numpy().copy())
            # compute output
            output= model(input_var)
            if len(output.shape)<2:
                output = torch.unsqueeze(output,0)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1,5))
            video_pred.extend(list(np.argmax(output.cpu().numpy().copy(), axis=1)))       

            #losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            print(output)
            log.write(output + '\n')
            log.flush()
    output = 'Testing costs {} mins {} seconds'.format(int(batch_time.sum)//60, int(batch_time.sum)%60)
    print(output)
    log.write(output + '\n')
    log.flush()
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt
    output=('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    print(output)
    print(cls_acc)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    if args.evaluate:
        return 
    writer.add_scalar('test/loss', losses.avg, epoch + 1)
    writer.add_scalar('test/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('test/top5Accuracy', top5.avg, epoch + 1)
    log.write(output + ' ' + output_best + '\n')
    log.flush()
    log.write(str(cls_acc)+'\n')
    log.flush()
    return top1.avg, cls_acc

def get_optim_policies(model):
    first_conv_weight = []
    first_conv_bias = []
    linear_weight = []
    linear_bias = []
    conv_weight=[]
    conv_bias=[]
    ParameterList_weight=[]
    diff_weight=[]
    diff_bias=[]
    bn = []

    conv_cnt = 0
    for name, m in model.named_modules():
        if 'diff' in name:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.BatchNorm3d):
                ps = list(m.parameters())
                diff_weight.append(ps[0])
                if len(ps) == 2:
                    diff_bias.append(ps[1])
        else:
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    conv_weight.append(ps[0])
                    if len(ps) == 2:
                        conv_bias.append(ps[1])
            elif isinstance(m, torch.nn.ParameterList):
                ParameterList_weight.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                linear_weight.append(ps[0])
                if len(ps) == 2:
                    linear_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):  # enable BN
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

    return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "conv2d_weight"},
            {'params': conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "conv2d_bias"},
            {'params': ParameterList_weight, 'lr_mult': 1, 'decay_mult': 0,
             'name': "ParameterList_weight"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': diff_weight, 'lr_mult': 5, 'decay_mult': 1,  #5 1
             'name': "lateral_weight"},
            {'params': diff_bias, 'lr_mult': 10, 'decay_mult': 0,  #10 0
             'name': "lateral_bias"},
        ]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def save_checkpoint(state, is_best, model_dir, epoch=0):
    if is_best:
        torch.save(state, '%s/%s_best.pth.tar' % (model_dir, args.store_name))
    else:
        torch.save(state, '%s/%s_checkpoint_%s.pth.tar' % (model_dir, args.store_name, str(epoch)))

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
