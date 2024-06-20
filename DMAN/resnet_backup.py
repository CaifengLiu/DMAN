import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from torch.cuda import FloatTensor as ftens
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
point=0
save_file=[]

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,lateral_layer=None,lateral_connection=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        self.stride = stride
        self.lateral_layer=lateral_layer
        self.lateral_connection=lateral_connection
            

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.lateral_layer!=None:
            out=self.lateral_connection(out,self.lateral_layer)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size=224,
                 shortcut_type='B',
                 num_classes=400,
                 num_segment=8):
        self.inplanes = 64
        self.num_segment=num_segment
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=(2, 2),
            padding=(3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool2d(
            ( last_size, last_size), stride=1)
        #self.dp=torch.nn.Dropout(0.6)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # V1 两个侧连接，tcover=2，用bottleneck卷积过去之后，作时间偏移两次，作平均之后和原时间的特征相加
        '''
        self.inplanes=512
        self.lateral_layer2=self._make_layer(block, 128, 1, shortcut_type)
        self.inplanes=1024
        self.lateral_layer3=self._make_layer(block, 256, 1, shortcut_type)
        self.t_cover=2
        '''
        #V2 两个侧连接，用部分通道，如1/8，用bottleneck作卷积之后，作时间偏移两次，替代原先时间的特征
        '''
        self.beta=1/8.0
        self.inplanes=int(512*self.beta)
        self.lateral_layer3=self._make_layer(block, int(128*self.beta), 1, shortcut_type)
        self.inplanes=int(1024*self.beta)
        self.lateral_layer4=self._make_layer(block, int(256*self.beta), 1, shortcut_type)
        self.t_cover=2
        '''
        # V3 在layer3和layer4的每个bottleneck都进行 1/8通道的输入，1/8通道的输出，作时间偏移两次，替代原先时间的特征的操作。
        '''
        self.beta=1/8.0
        self.inplanes=int(256*self.beta)
        self.lateral_layer3=self._make_layer(block, int(64*self.beta), 1, shortcut_type)
        self.inplanes=int(512*self.beta)
        self.lateral_layer4=self._make_layer(block, int(128*self.beta), 1, shortcut_type)
        self.t_cover=2
        self.inplanes=512
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, lateral_layer=self.lateral_layer3)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, lateral_layer=self.lateral_layer4)
        '''
        # V4 V5 V6 V8 V9用上一时间的特征减去这一时间的特征，用残差进行学习，然后加到这一时间的特征上
        '''
        self.beta=1/8.0
        self.inplanes=int(512*self.beta)
        self.lateral_layer3=self._make_layer(block, int(128*self.beta), 1, shortcut_type)
        self.inplanes=int(1024*self.beta)
        self.lateral_layer4=self._make_layer(block, int(256*self.beta), 1, shortcut_type)
        '''
        # V7 None
        #V10
        
        self.beta=1/8.0
        self.inplanes=int(512*self.beta)
        self.lateral_layer3=self._make_layer(block, int(128*self.beta), 1, shortcut_type)
        self.backward_layer3=self._make_layer(block, int(128*self.beta), 1, shortcut_type)
        self.inplanes=int(1024*self.beta)
        self.lateral_layer4=self._make_layer(block, int(256*self.beta), 1, shortcut_type)
        self.backward_layer4=self._make_layer(block, int(256*self.beta), 1, shortcut_type)
        
        #V11
        self.beta=1/8.0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, lateral_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,lateral_layer=lateral_layer,lateral_connection=self.lateral_connection))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,lateral_layer=lateral_layer,lateral_connection=self.lateral_connection))

        return nn.Sequential(*layers)

    def lateral_connection(self, input, lateral_layer, backward_layer=None):
        # V1
        '''
        nt, c ,h ,w= input.size()
        output=lateral_layer(input)
        output=output.view(nt//self.num_segment,self.num_segment,c, h, w)
        
        output=torch.cat((ftens(output.size(0),  1, c, h, w).fill_(0), output[:,:-1]), dim=1)
        output_1=torch.cat((ftens(output.size(0),  1, c, h, w).fill_(0), output[:,:-1]), dim=1)
        
        input=input.view(input.size(0)//self.num_segment,self.num_segment,c, h, w)+(output+output_1)/2
        return input.view(nt,c, h, w)
        '''
        #V2 V3
        '''
        nt, c ,h ,w= input.size()
        channels=int(c*self.beta)
        output=input[:,:channels]
        output=lateral_layer(output)
        output=output.view(input.size(0)//self.num_segment,self.num_segment,c, h, w)
        output=torch.cat((ftens(output.size(0),  1, c, h, w).fill_(0), output[:,:-1]), dim=1)
        output_1=torch.cat((ftens(output.size(0),  1, c, h, w).fill_(0), output[:,:-1]), dim=1)
        
        input=input.view(input.size(0)//self.num_segment,self.num_segment,c, h, w)
        input=torch.cat((input[:,:,channels:],(output+output_1)/2 ),2)
        return input.view(nt,c, h, w)
        '''
        # V4 V5 V6 V8
        '''
        nt, c ,h ,w= input.size()
        channels=int(c*self.beta)
        input=input.view(nt//self.num_segment,self.num_segment,c, h, w)
        output=torch.zeros_like(input[:,:,:channels])
        #output[:,1:]=input[:,:-1,:channels]-input[:,1:,:channels]
        #V8 uses this line
        output[:,1:]=input[:,:-1,channels:2*channels]-input[:,1:,channels:2*channels]
        output=output.view(nt,channels, h, w)
        output=lateral_layer(output)
        output=torch.cat((output,torch.zeros(nt,c-channels,h,w).cuda()),1)
        input=input.view(nt,c, h, w)
        return input+output
        '''
        #V9 
        '''
        nt, c ,h ,w= input.size()
        channels=int(c*self.beta)
        input=input.view(nt//self.num_segment,self.num_segment,c, h, w)
        output=torch.zeros_like(input[:,:,:channels])
        output[:,1:]=input[:,:-1,2*channels:3*channels]-input[:,1:,2*channels:3*channels]
        output=output.view(nt,channels, h, w)
        output=lateral_layer(output)
        input=input.view(nt,c, h, w)
        out=torch.cat((output,input[:,channels:]),1)
        return out
        '''
        #V10
        '''
        nt, c ,h ,w= input.size()
        channels=int(c*self.beta)
        input=input.view(nt//self.num_segment,self.num_segment,c, h, w)
        output=torch.zeros_like(input[:,:,:channels])
        output_back=torch.zeros_like(input[:,:,:channels])
        output[:,1:]=input[:,:-1,2*channels:3*channels]-input[:,1:,2*channels:3*channels]
        output_back[:,:-1]=input[:,1:,2*channels:3*channels]-input[:,:-1,2*channels:3*channels]
        output=output.view(nt,channels, h, w)
        output_back=output_back.view(nt,channels, h, w)
        output=lateral_layer(output)
        output_back=backward_layer(output_back)
        input=input.view(nt,c, h, w)
        out=torch.cat((output,output_back,input[:,channels*2:]),1)
        return out
        '''
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # V1 V2 V4 V5 V6 V7 V8 V9
        #x = self.lateral_connection(x,self.lateral_layer3)
        x = self.lateral_connection(x,self.lateral_layer3,self.backward_layer3)
        x = self.layer3(x)
        # V1 V2 V4 V5 V6 V7 V8 V9
        #x = self.lateral_connection(x,self.lateral_layer4)
        x = self.lateral_connection(x,self.lateral_layer4,self.backward_layer4)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = self.dp(x)
        x = x.view(x.size(0), -1)
        '''
        global point,save_file
        if(point == 0):
            if len(x.size())<2:
                save_file=torch.unsqueeze(x,dim=0).cpu().numpy()
            else:
                save_file=x.cpu().numpy()
        else:
            if len(x.size())<2:
                save_file=np.concatenate((save_file,x.unsqueeze(0).cpu().numpy()),axis=0)
            else:
                save_file=np.concatenate((save_file,x.cpu().numpy()),axis=0)
        point+=x.size()[0]
        print(point)
        if(point >= 425):       #989 425
            np.save('features_res101_openpose_crop_val_0',save_file)
            print(save_file.shape)
        '''
        x = self.fc(x)
        
        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
