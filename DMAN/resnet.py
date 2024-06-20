import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet50', 'resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, r=1/8):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d( planes, planes, kernel_size=(1,3,3), stride=(1,stride,stride), padding=(0,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, int(planes * self.expansion * (1-r*2)), kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(int(planes * self.expansion * (1-r*2)))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        div_channel = int(planes * self.expansion *r)
        self.diff_map1 = nn.Sequential(
            nn.Conv3d(planes, div_channel, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(div_channel),
            nn.LeakyReLU(0.1),
        )
        self.diff_map2 = nn.Sequential(
            nn.Conv3d(planes, div_channel, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(div_channel),
            nn.LeakyReLU(0.1),
        )

        #choice2 
        #div_channel = int(planes * self.expansion *(1-2*r))
        #choice3
        '''
        div_channel = 1
        self.diff_map = nn.Conv3d(div_channel, int(planes * self.expansion *(1-2*r)), kernel_size=1)
        self.diff_weight1_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,1,1), groups=div_channel)
        self.diff_weight2_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=div_channel)
        self.diff_weight3_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=div_channel)
        self.diff_weight4_forward =  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.diff_weight1_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,1,1), groups=div_channel)
        self.diff_weight2_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=div_channel)
        self.diff_weight3_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=div_channel)
        self.diff_weight4_backward =  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.weights = nn.Parameter(torch.ones(8) / 8, requires_grad=True)
        '''
        #choice4
        '''
        div_channel = 4
        self.dif1_transform = nn.Conv3d(planes, div_channel, kernel_size=(1,1,1))
        self.dif2_transform = nn.Conv3d(planes, div_channel, kernel_size=(1,1,1))
        self.diff_weight1_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,1,1), groups=div_channel)
        self.diff_weight2_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=div_channel)
        #self.diff_weight3_forward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=div_channel)
        self.diff_weight4_forward =  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.diff_weight1_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,1,1), groups=div_channel)
        self.diff_weight2_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=div_channel)
        #self.diff_weight3_backward = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=div_channel)
        self.diff_weight4_backward =  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.weights = nn.Parameter(torch.ones(8) / 6, requires_grad=True)
        self.dif_transform_back = nn.Conv3d(div_channel, planes, kernel_size=(1,1,1))
        '''
        #choice5
        div_channel = 4
        self.diff_transform = nn.Conv3d(planes, div_channel, kernel_size=(1,1,1))
        self.diff_weight1 = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,1,1), groups=div_channel)
        self.diff_weight2 = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,2,2), dilation=(1,2,2), groups=div_channel)
        self.diff_weight3 = nn.Conv3d(div_channel, div_channel, kernel_size=(1,3,3), padding=(0,3,3), dilation=(1,3,3), groups=div_channel)
        self.diff_weight4 =  nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.weights = nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        self.diff_transform_back1 = nn.Conv3d(div_channel, planes, kernel_size=(1,1,1))
        self.diff_transform_back2 = nn.Conv3d(div_channel, planes, kernel_size=(1,1,1))

    def forward(self, x):
        residual = x
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        N, C, T, H, W = out.size()
        dif1 = out[:, :, 1:] - out[:, :, :-1]
        dif1 = torch.cat([dif1.new(N, C, 1, H, W).zero_(), dif1], dim=2)
        dif2 = out[:, :, :-1] - out[:, :, 1:]
        dif2 = torch.cat([dif2, dif2.new(N, C, 1, H, W).zero_()], dim=2)
        #choice4
        '''
        out_dif1 = self.dif1_transform(dif1)
        out_dif2 = self.dif1_transform(dif2)
        diff_weight = self.diff_weight1_forward(out_dif1)*self.weights[0] + self.diff_weight2_forward(out_dif1)*self.weights[1] \
                    + F.interpolate(self.diff_weight4_forward(out_dif1), out_dif1.size()[2:])*self.weights[3]\
                    + self.diff_weight1_backward(out_dif2)*self.weights[4] + self.diff_weight2_backward(out_dif2)*self.weights[5] \
                    + F.interpolate(self.diff_weight4_backward(out_dif2), out_dif2.size()[2:])*self.weights[7]
                    #+ self.diff_weight3_forward(out_dif1)*self.weights[2] 
                    #+ self.diff_weight3_backward(out_dif2)*self.weights[6] 
        out = out + out*self.dif_transform_back(diff_weight)
        '''
        #choice5
        out_dif = self.diff_transform(out)
        N, C, T, H, W = out_dif.size()
        diff_weight = self.diff_weight1(out_dif)*self.weights[0] + self.diff_weight2(out_dif)*self.weights[1] \
                    + self.diff_weight3(out_dif)*self.weights[2] + F.interpolate(self.diff_weight4(out_dif), out_dif.size()[2:])*self.weights[3]
        out_dif1 = diff_weight[:, :, 1:] - out_dif[:, :, :-1]
        out_dif1 = torch.cat([out_dif1.new(N, C, 1, H, W).zero_(), out_dif1], dim=2)
        out_dif2 = diff_weight[:, :, :-1] - out_dif[:, :, 1:]
        out_dif2 = torch.cat([out_dif2, out_dif2.new(N, C, 1, H, W).zero_()], dim=2)
        out_diff_weight1 = self.diff_transform_back1(out_dif1)
        out_diff_weight2 = self.diff_transform_back2(out_dif2)
        out = out + out*((F.sigmoid(out_diff_weight1)+F.sigmoid(out_diff_weight2))/2-0.5)
        
        dif1 = self.diff_map1(dif1)
        dif2 = self.diff_map2(dif2)
        out = self.bn3(self.conv3(out))
        out = torch.cat((out, dif1, dif2), 1)
        
        #dif1 = self.diff_map1(dif1)
        #dif2 = self.diff_map2(dif2)
        #N, C, T, H, W = out.size()
        #out_dif1 = torch.cat([out.new(N, C, 1, H, W).zero_(), out[:, :, 1:] - out[:, :, :-1]], dim=2)
        #out_dif2 = torch.cat([out[:, :, :-1] - out[:, :, 1:], out.new(N, C, 1, H, W).zero_()], dim=2)
        #diff_weight = self.diff_weight1_forward(out_dif1)*self.weights[0] + self.diff_weight2_forward(out_dif1)*self.weights[1] \
        #            + self.diff_weight3_forward(out_dif1)*self.weights[2] + F.interpolate(self.diff_weight4_forward(out_dif1), out_dif1.size()[2:])*self.weights[3]\
        #            + self.diff_weight1_backward(out_dif2)*self.weights[4] + self.diff_weight2_backward(out_dif2)*self.weights[5] \
        #            + self.diff_weight3_backward(out_dif2)*self.weights[6] + F.interpolate(self.diff_weight4_backward(out_dif2), out_dif2.size()[2:])*self.weights[7]
        #out = torch.cat((out+out*self.diff_map(diff_weight), dif1, dif2), 1)
        
        #choice2
        #dif1 = self.diff_map1(dif1)
        #dif2 = self.diff_map2(dif2)
        #diff_weight = self.diff_weight1_forward(dif1)*self.weights[0] + self.diff_weight2_forward(dif1)*self.weights[1] \
        #            + self.diff_weight3_forward(dif1)*self.weights[2] + F.interpolate(self.diff_weight4_forward(dif1), dif1.size()[2:])*self.weights[3]\
        #            + self.diff_weight1_backward(dif2)*self.weights[4] + self.diff_weight2_backward(dif2)*self.weights[5] \
        #            + self.diff_weight3_backward(dif2)*self.weights[6] + F.interpolate(self.diff_weight4_backward(dif2), dif2.size()[2:])*self.weights[7]
        #out = torch.cat((out+out*self.diff_map(diff_weight), dif1, dif2), 1)
        
        #choice3
        #dif1 = self.diff_map1(dif1)
        #dif2 = self.diff_map2(dif2)
        #out_dif1 = dif1.mean(1, keepdim=True)
        #out_dif2 = dif2.mean(1, keepdim=True)
        #diff_weight = self.diff_weight1_forward(out_dif1)*self.weights[0] + self.diff_weight2_forward(out_dif1)*self.weights[1] \
        #            + self.diff_weight3_forward(out_dif1)*self.weights[2] + F.interpolate(self.diff_weight4_forward(out_dif1), out_dif1.size()[2:])*self.weights[3]\
        #            + self.diff_weight1_backward(out_dif2)*self.weights[4] + self.diff_weight2_backward(out_dif2)*self.weights[5] \
        #            + self.diff_weight3_backward(out_dif2)*self.weights[6] + F.interpolate(self.diff_weight4_backward(out_dif2), out_dif2.size()[2:])*self.weights[7]
        #out = torch.cat((out+out*self.diff_map(diff_weight), dif1, dif2), 1)
        

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, resi=False, beta=1/8.0):
        self.inplanes = 64
        self.beta = beta
        super(ResNet, self).__init__()
        self.diff_map1 = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(0.1),
        )
        self.diff_map2 = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1)),
            nn.BatchNorm3d(3),
            nn.LeakyReLU(0.1),
        )
        self.conv1 = nn.Conv3d(12, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0], r=self.beta)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, r=self.beta)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, r=self.beta)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, r=self.beta)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.resi=resi
        '''
        if resi:
            
            #self.inplanes=int(256*self.beta)
            #self.lateral_layer2=self._make_layer(block, int(64*self.beta), 1)
            #self.backward_layer2=self._make_layer(block, int(64*self.beta), 1)
            self.inplanes=int(512*self.beta)
            self.lateral_layer3=self._make_layer(block, int(128*self.beta), 1)
            self.backward_layer3=self._make_layer(block, int(128*self.beta), 1)
            self.inplanes=int(1024*self.beta)
            self.lateral_layer4=self._make_layer(block, int(256*self.beta), 1)
            self.backward_layer4=self._make_layer(block, int(256*self.beta), 1)
        '''
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, r=1/8.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, r=r))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, r=r))

        return nn.Sequential(*layers)

    def lateral_connection(self, input, lateral_layer, backward_layer=None):
        
        n, c, t ,h ,w= input.size()
        channels=int(c*self.beta)
        output=torch.zeros_like(input[:,:channels])
        output_back=torch.zeros_like(input[:,:channels])
        output[:,:,1:]=input[:,2*channels:3*channels,:-1]-input[:,2*channels:3*channels,1:]
        output_back[:,:,:-1]=input[:,2*channels:3*channels,1:]-input[:,2*channels:3*channels,:-1]
        output=lateral_layer(output)
        output_back=backward_layer(output_back)
        out=torch.cat((output,output_back,input[:,channels*2:]),1)
        return out
    def forward(self, x):
        # ncthw
        N, C, T, H, W = x.size()
        #x = x.reshape(N//3, 3, C, T, H, W)
        #dif1 = x[:, 1] - x[:, 0]
        dif1 = x[:, :, 1:] - x[:, :, 0:-1]
        dif1 = torch.cat([dif1.new(N, C, 1, H, W).zero_(), dif1], dim=2)
        #dif2 = x[:, 2] - x[:, 1]
        dif2 = x[:, :, :-1] - x[:, :, 1:]
        dif2 = torch.cat([dif2, dif2.new(N, C, 1, H, W).zero_()], dim=2)
        #x = x[:,1]
        #x = x+ self.diff_map1(dif1)+ self.diff_map2(dif2)
        x = torch.cat((x, x, self.diff_map1(dif1), self.diff_map2(dif2)), dim = 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #if self.resi:
        #    x = self.lateral_connection(x,self.lateral_layer2,self.backward_layer2)
        x = self.layer2(x)
        #if self.resi:
        #    x = self.lateral_connection(x,self.lateral_layer3,self.backward_layer3)
        x = self.layer3(x)
        #if self.resi:
        #    x = self.lateral_connection(x,self.lateral_layer4,self.backward_layer4)
        x = self.layer4(x)
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 based model.
    """
    beta = 1/8.0
    model = ResNet(Bottleneck, [3, 4, 6, 3], beta = beta, **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'lateral' in layer_name:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
        if 'conv3' in ln:
            n_out, n_in, _, _, _ = checkpoint[ln].size()
            checkpoint[ln.replace('conv3','diff_map1.0')] = checkpoint[ln][int(n_out*(1-beta*2)):int(n_out*(1-beta))].repeat(1,1,1,3,3)/9
            checkpoint[ln.replace('conv3','diff_map2.0')] = checkpoint[ln][int(n_out*(1-beta)):].repeat(1,1,1,3,3)/9
            checkpoint[ln] = checkpoint[ln][:int(n_out*(1-beta*2))]
        elif 'bn3' in ln:
            n_out = checkpoint[ln].size()[0]
            checkpoint[ln.replace('bn3','diff_map1.1')] = checkpoint[ln][int(n_out*(1-beta*2)):int(n_out*(1-beta))]
            checkpoint[ln.replace('bn3','diff_map2.1')] = checkpoint[ln][int(n_out*(1-beta)):]
            checkpoint[ln] = checkpoint[ln][:int(n_out*(1-beta*2))]
        if ln == 'conv1.weight':
            checkpoint[ln] = checkpoint[ln].repeat(1,4,1,1,1)/4
    model.load_state_dict(checkpoint,strict = False)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        groups
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'lateral' in layer_name:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint,strict = False)
    return model
