import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import BatchNorm2d
from torch.nn import SyncBatchNorm as BatchNorm2d
from functools import partial
import re
from models.base_models.resnet import resnet101, resnet18, resnet50
from utils.seg_opr.conv_2_5d import Conv2_5D_depth, Conv2_5D_disp

class DeepLabV3p_r18(nn.Module):
    def __init__(self, num_classes, config):
        super(DeepLabV3p_r18, self).__init__()
        self.norm_layer = BatchNorm2d
        self.backbone = resnet18(config.root_dir+'/data/pytorch-weight/resnet18_v1c.pth', norm_layer=self.norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=False, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head('r18', num_classes, self.norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)
        init_weight(self.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.classifier, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def forward(self, data, get_sup_loss = False, gt = None, criterion = None):
        data = data[0] #rgb is the first element in the list
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      #(b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        if not self.training: #return pred for evaluation
            return pred
        else:
            if get_sup_loss:
                return pred, self.get_sup_loss(pred, gt, criterion)
            else:
                return pred
            
    def get_sup_loss(self, pred, gt, criterion):
        pred = pred[:gt.shape[0]] #Getting loss for only those examples in batch where gt exists. Won't get sup loss for unlabeled data. 
        return criterion(pred, gt)

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def get_params(self):
        param_groups = [[], [], []]
        enc, enc_no_decay = group_weight(self.backbone, self.norm_layer)
        param_groups[0].extend(enc)
        param_groups[1].extend(enc_no_decay)
        dec, dec_no_decay = group_weight(self.head, self.norm_layer)
        param_groups[2].extend(dec)
        param_groups[1].extend(dec_no_decay)
        classifier, classifier_no_decay = group_weight(self.classifier, self.norm_layer)
        param_groups[2].extend(classifier)
        param_groups[1].extend(classifier_no_decay)
        return param_groups

class DeepLabV3p_r50(nn.Module):
    def __init__(self, num_classes, config):
        super(DeepLabV3p_r50, self).__init__()
        self.norm_layer = BatchNorm2d
        self.backbone = resnet50(config.root_dir+'/data/pytorch-weight/resnet50_v1c.pth', norm_layer=self.norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head('r50', num_classes, self.norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)
        init_weight(self.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.classifier, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')


    def forward(self, data, get_sup_loss = False, gt = None, criterion = None):
        data = data[0] #rgb is the first element in the list
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      #(b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        if not self.training: #return pred for evaluation
            return pred
        else:
            if get_sup_loss:
                return pred, self.get_sup_loss(pred, gt, criterion)
            else:
                return pred
            
    def get_sup_loss(self, pred, gt, criterion):
        pred = pred[:gt.shape[0]] #Getting loss for only those examples in batch where gt exists. Won't get sup loss for unlabeled data. 
        return criterion(pred, gt)

    def get_params(self):
        param_groups = [[], [], []]
        enc, enc_no_decay = group_weight(self.backbone, self.norm_layer)
        param_groups[0].extend(enc)
        param_groups[1].extend(enc_no_decay)
        dec, dec_no_decay = group_weight(self.head, self.norm_layer)
        param_groups[2].extend(dec)
        param_groups[1].extend(dec_no_decay)
        classifier, classifier_no_decay = group_weight(self.classifier, self.norm_layer)
        param_groups[2].extend(classifier)
        param_groups[1].extend(classifier_no_decay)
        return param_groups
    
    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

class DeepLabV3p_r101(nn.Module):
    def __init__(self, num_classes, config):
        super(DeepLabV3p_r101, self).__init__()
        self.norm_layer = BatchNorm2d
        self.backbone = resnet101(config.root_dir+'/data/pytorch-weight/resnet101_v1c.pth', norm_layer=self.norm_layer,
                                  bn_eps=config.bn_eps,
                                  bn_momentum=config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.head = Head('r50', num_classes, self.norm_layer, config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)
        init_weight(self.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        init_weight(self.classifier, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')


    def forward(self, data, get_sup_loss = False, gt = None, criterion = None):
        data = data[0] #rgb is the first element in the list
        blocks = self.backbone(data)
        v3plus_feature = self.head(blocks)      #(b, c, h, w)
        b, c, h, w = v3plus_feature.shape

        pred = self.classifier(v3plus_feature)

        b, c, h, w = data.shape
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
        if not self.training: #return pred for evaluation
            return pred
        else:
            if get_sup_loss:
                return pred, self.get_sup_loss(pred, gt, criterion)
            else:
                return pred
            
    def get_sup_loss(self, pred, gt, criterion):
        pred = pred[:gt.shape[0]] #Getting loss for only those examples in batch where gt exists. Won't get sup loss for unlabeled data. 
        return criterion(pred, gt)

    def get_params(self):
        param_groups = [[], [], []]
        enc, enc_no_decay = group_weight(self.backbone, self.norm_layer)
        param_groups[0].extend(enc)
        param_groups[1].extend(enc_no_decay)
        dec, dec_no_decay = group_weight(self.head, self.norm_layer)
        param_groups[2].extend(dec)
        param_groups[1].extend(dec_no_decay)
        classifier, classifier_no_decay = group_weight(self.classifier, self.norm_layer)
        param_groups[2].extend(classifier)
        param_groups[1].extend(classifier_no_decay)
        return param_groups
    
    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
        pool = pool.view(x.size(0), x.size(1), 1, 1)
        return pool


class Head(nn.Module):
    def __init__(self, base_model, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        if base_model == 'r18':
            self.aspp = ASPP(512, 256, [6, 12, 18], norm_act=norm_act)
            
            self.reduce = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
            )
        elif base_model == 'r50':
            self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)
            self.reduce = nn.Sequential(
                nn.Conv2d(256, 48, 1, bias=False),
                norm_act(48, momentum=bn_momentum),
                nn.ReLU(),
            )
        else:
            raise Exception(f"Head not implemented for {base_model}")


        
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f
    

def group_weight(module, norm_layer):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, Conv2_5D_depth):
                group_decay.append(m.weight_0)
                group_decay.append(m.weight_1)
                group_decay.append(m.weight_2)
                if m.bias is not  None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, Conv2_5D_disp):
                group_decay.append(m.weight_0)
                group_decay.append(m.weight_1)
                group_decay.append(m.weight_2)
                if m.bias is not  None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.Parameter):
                group_decay.append(m)
            elif isinstance(m, nn.Embedding):
                group_decay.append(m)
        assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
        return group_decay, group_no_decay

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, Conv2_5D_depth):
            conv_init(m.weight_0, **kwargs)
            conv_init(m.weight_1, **kwargs)
            conv_init(m.weight_2, **kwargs)
        elif isinstance(m, Conv2_5D_disp):
            conv_init(m.weight_0, **kwargs)
            conv_init(m.weight_1, **kwargs)
            conv_init(m.weight_2, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)