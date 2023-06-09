import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import SyncBatchNorm as BatchNorm2d
import re
import os, sys
from six import moves

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), BatchNorm2d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class MyRefineNet(nn.Module):
    def __init__(self, num_layers, num_classes):
        super(MyRefineNet, self).__init__()
        self.model = refinenet(num_layers, num_classes, 1, None) #Passing num_parallel = 1 and bn_threshold as None
        self.model = model_init(self.model, num_layers, 1, imagenet=True) #Only initializes the encoder
    def forward(self, data, get_sup_loss = False, gt = None, criterion = None):
        b, c, h, w = data[0].shape  #rgb is the 0th element
        pred = self.model(data)
        pred = F.interpolate(pred[0], size=(h, w), mode='bilinear', align_corners=True)
        if not self.training: # return just predictions for evaluation
            return pred
        else:
            if get_sup_loss:
                return pred, self.get_sup_loss(pred, gt, criterion)
            else:
                return pred

    def get_params(self):
        enc_params, dec_params= [], []
        for name, param in self.model.named_parameters():
            if bool(re.match('.*conv1.*|.*bn1.*|.*layer.*', name)):
                enc_params.append(param)
            else:
                dec_params.append(param)
        return enc_params, dec_params

    def get_sup_loss(self, pred, gt, criterion):
        pred = pred[:gt.shape[0]] #Getting loss for only those examples in batch where gt exists. Won't get sup loss for unlabeled data. 
        return criterion(pred, gt)

"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


models_urls = {
    '101_voc'     : 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',

    '18_imagenet' : 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    '50_imagenet' : 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

bottleneck_idx = 0
save_idx = 0


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))


class CRPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_stages, num_parallel):
        super(CRPBlock, self).__init__()
        for i in range(num_stages):
            setattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                    conv3x3(in_planes if (i == 0) else out_planes, out_planes))
        self.stride = 1
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=5, stride=1, padding=2))

    def forward(self, x):
        top = x
        for i in range(self.num_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = [x[l] + top[l] for l in range(self.num_parallel)]
        return x


stages_suffixes = {0 : '_conv', 1 : '_conv_relu_varout_dimred'}

class RCUBlock(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, num_stages, num_parallel):
        super(RCUBlock, self).__init__()
        for i in range(num_blocks):
            for j in range(num_stages):
                setattr(self, '{}{}'.format(i + 1, stages_suffixes[j]),
                        conv3x3(in_planes if (i == 0) and (j == 0) else out_planes,
                                out_planes, bias=(j == 0)))
        self.stride = 1
        self.num_blocks = num_blocks
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
    
    def forward(self, x):
        for i in range(self.num_blocks):
            residual = x
            for j in range(self.num_stages):
                x = self.relu(x)
                x = getattr(self, '{}{}'.format(i + 1, stages_suffixes[j]))(x)
            x = [x[l] + residual[l] for l in range(self.num_parallel)]
        return x


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, block, layers, num_parallel, num_classes=21, bn_threshold=2e-2):
        self.inplanes = 64
        self.num_parallel = num_parallel
        super(RefineNet, self).__init__()
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False))
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)

        self.p_ims1d2_outl1_dimred = conv3x3(2048, 512)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(512, 256)

        self.p_ims1d2_outl2_dimred = conv3x3(1024, 256)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(256, 256)

        self.p_ims1d2_outl3_dimred = conv3x3(512, 256)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(256, 256)

        self.p_ims1d2_outl4_dimred = conv3x3(256, 256)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(256, 256)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g4_b = self._make_rcu(256, 256, 3, 2)

        self.clf_conv = conv3x3(256, num_classes, bias=True)

    def _make_crp(self, in_planes, out_planes, num_stages):
        layers = [CRPBlock(in_planes, out_planes, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, num_blocks, num_stages):
        layers = [RCUBlock(in_planes, out_planes, num_blocks, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout(l4)
        l3 = self.dropout(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = [nn.Upsample(size=l3[0].size()[2:], mode='bilinear', align_corners=True)(x4_) for x4_ in x4]

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = [x3[l] + x4[l] for l in range(self.num_parallel)]
        x3 = self.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = [nn.Upsample(size=l2[0].size()[2:], mode='bilinear', align_corners=True)(x3_) for x3_ in x3]

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = [x2[l] + x3[l] for l in range(self.num_parallel)]
        x2 = self.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = [nn.Upsample(size=l1[0].size()[2:], mode='bilinear', align_corners=True)(x2_) for x2_ in x2]

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = [x1[l] + x2[l] for l in range(self.num_parallel)]
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.dropout(x1)

        out = self.clf_conv(x1)
        return out


class RefineNet_Resnet18(nn.Module):
    def __init__(self, block, layers, num_parallel, num_classes=21, bn_threshold=2e-2):
        self.inplanes = 64
        self.num_parallel = num_parallel
        super(RefineNet_Resnet18, self).__init__()
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  bias=False))
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)

        self.p_ims1d2_outl1_dimred = conv3x3(512, 256)
        self.adapt_stage1_b = self._make_rcu(256, 256, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g1_b = self._make_rcu(256, 256, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3(256, 64)

        self.p_ims1d2_outl2_dimred = conv3x3(256, 64)
        self.adapt_stage2_b = self._make_rcu(64, 64, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3(64, 64)
        self.mflow_conv_g2_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g2_b = self._make_rcu(64, 64, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3(64, 64)

        self.p_ims1d2_outl3_dimred = conv3x3(128, 64)
        self.adapt_stage3_b = self._make_rcu(64, 64, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3(64, 64)
        self.mflow_conv_g3_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g3_b = self._make_rcu(64, 64, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3(64, 64)

        self.p_ims1d2_outl4_dimred = conv3x3(64, 64)
        self.adapt_stage4_b = self._make_rcu(64, 64, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3(64, 64)
        self.mflow_conv_g4_pool = self._make_crp(64, 64, 4)
        self.mflow_conv_g4_b = self._make_rcu(64, 64, 3, 2)

        self.clf_conv = conv3x3(64, num_classes, bias=True)

    def _make_crp(self, in_planes, out_planes, num_stages):
        layers = [CRPBlock(in_planes, out_planes, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)
    
    def _make_rcu(self, in_planes, out_planes, num_blocks, num_stages):
        layers = [RCUBlock(in_planes, out_planes, num_blocks, num_stages, self.num_parallel)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.dropout(l4)
        l3 = self.dropout(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = [nn.Upsample(size=l3[0].size()[2:], mode='bilinear', align_corners=True)(x4_) for x4_ in x4]

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = [x3[l] + x4[l] for l in range(self.num_parallel)]
        x3 = self.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = [nn.Upsample(size=l2[0].size()[2:], mode='bilinear', align_corners=True)(x3_) for x3_ in x3]

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = [x2[l] + x3[l] for l in range(self.num_parallel)]
        x2 = self.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = [nn.Upsample(size=l1[0].size()[2:], mode='bilinear', align_corners=True)(x2_) for x2_ in x2]

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = [x1[l] + x2[l] for l in range(self.num_parallel)]
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = self.dropout(x1)

        out = self.clf_conv(x1)
        
        return out


def refinenet(num_layers, num_classes, num_parallel, bn_threshold):
    refinnetClass = RefineNet
    if int(num_layers) == 18:
        layers = [2, 2, 2, 2]
        block = BasicBlock
        refinnetClass = RefineNet_Resnet18
    elif int(num_layers) == 50:
        layers = [3, 4, 6, 3]
        block = Bottleneck
    elif int(num_layers) == 101:
        layers = [3, 4, 23, 3]
        block = Bottleneck
    elif int(num_layers) == 152:
        layers = [3, 8, 36, 3]
        block = Bottleneck
    else:
        print('invalid num_layers')

    model = refinnetClass(block, layers, num_parallel, num_classes, bn_threshold)
    return model

def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        moves.urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

def model_init(model, num_layers, num_parallel, imagenet=False):
    if imagenet:
        key = str(num_layers) + '_imagenet'
        url = models_urls[key]
        state_dict = maybe_download(key, url)
        model_dict = expand_model_dict(model.state_dict(), state_dict, num_parallel)
        model.load_state_dict(model_dict, strict=True)
    return model

def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict        

def get_params(model):
    enc_params, dec_params, slim_params = [], [], []
    for name, param in model.named_parameters():
        if bool(re.match('.*conv1.*|.*bn1.*|.*layer.*', name)):
            enc_params.append(param)
            # if args.print_network:
            # print(' Enc. parameter: {}'.format(name))
        else:
            dec_params.append(param)
    #         if args.print_network:
            # print(' Dec. parameter: {}'.format(name))
        if param.requires_grad and name.endswith('weight') and 'bn2' in name:
            if len(slim_params) % 2 == 0:
                slim_params.append(param[:len(param) // 2])
            else:
                slim_params.append(param[len(param) // 2:])
    return enc_params, dec_params, slim_params

def get_sup_loss_from_output(criterion, outputs, target):
    target = target.squeeze(3)
    loss = 0
    for output in outputs:
        output = nn.functional.interpolate(output, size=target.size()[1:],
                                            mode='bilinear', align_corners=False)
        # soft_output = torch.nn.functional.log_softmax(output, dim = 1)
        # Compute loss and backpropagate
        # print(soft_output.shape, target.shape, "Shapes")
        # print(soft_output[0, :, 0, 0], target[0, 0, 0], "values")
        # loss += criterion(soft_output, target)
        # print(output[0, :, 0, 0], target[0, 0, 0], "values")
        loss += criterion(output, target)
    return loss/len(outputs)

def L1_penalty(var):
    return torch.abs(var).sum()#.to(var.get_device())