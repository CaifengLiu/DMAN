import torch.nn as nn
from transforms import *
from torch.nn.init import normal_, constant_



class TemporalModel(nn.Module):
    def __init__(self, num_class, num_segments, model, backbone, alpha, beta, 
            dropout = 0.3, resi=False):    
        super(TemporalModel, self).__init__()
        self.num_segments = num_segments
        self.model = model
        self.backbone = backbone[:-4]
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.resi=resi
        self._prepare_base_model(model,backbone[:-4])
        std = 0.001
        if self.dropout == 0:
            setattr(self.base_model,self.base_model.last_layer_name,nn.Linear(self.feature_dim,num_class))
            self.new_fc = None
            normal_(getattr(self.base_model,self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model,self.base_model.last_layer_name).bias, 0)
        else:
            setattr(self.base_model,self.base_model.last_layer_name,nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(self.feature_dim,num_class)
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)

        

    def _prepare_base_model(self, model, backbone):
        if model == 'GST':
            import GST
            self.base_model = getattr(GST,backbone)(alpha = self.alpha, beta = self.beta, resi=self.resi)
            print(' initialize GST with alpha: {} beta:{} resi:{} '.format(self.alpha,self.beta,self.resi,))
        elif model == 'STM':
            import STM
            self.base_model = getattr(STM,backbone)(resi=self.resi)
        elif model == 'TMP':
            import resnet_tmp
            self.base_model = getattr(resnet_tmp,backbone)(resi=self.resi)
            print(' initialize TMP with backbone: {} num_segments:{} resi:{} dropout:{}'.format(self.backbone,self.num_segments,self.resi,self.dropout))
        elif model == 'TSM':
            import resnet_tsm
            self.base_model = getattr(resnet_tsm,backbone)(resi=self.resi)
            print(' initialize TSM with backbone: {} num_segments:{} resi:{} dropout:{}'.format(self.backbone,self.num_segments,self.resi,self.dropout))
        elif model == 'ORI':
            import resnet
            self.base_model = getattr(resnet,backbone)(resi=self.resi)
            print(' initialize ORI with backbone: {} num_segments:{} resi:{} dropout:{}'.format(self.backbone,self.num_segments,self.resi,self.dropout))
        elif model == 'C3D':
            import C3D
            self.base_model = getattr(C3D,backbone)()
        elif model == 'S3D':
            import S3D
            self.base_model = getattr(S3D,backbone)()
        else:
            raise ValueError('Unknown model: {}'.format(model))
        
        if 'resnet' in backbone:
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.init_crop_size = 256
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if backbone == 'resnet18' or backbone == 'resnet34':
                self.feature_dim = 512
            else:
                self.feature_dim = 2048
        
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))


    def forward(self, input):
        #print(input.size()) torch.Size([16, 21, 224, 224])
        input = input.view((-1, self.num_segments,3) + input.size()[-2:])
        input = input.transpose(1,2).contiguous()
        base_out = self.base_model(input)
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        base_out = base_out.view((-1,self.num_segments)+base_out.size()[1:])
        output = base_out.mean(dim=1)

        return output

    
    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * self.init_crop_size // self.input_size

     