import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from loss.arcface import ArcFace
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.cvt import cvt_21_224_TransReID
from .backbones.cvt_uda import uda_cvt_21_224_TransReID
# from .backbones.t2t_vit import t2t_vit_14
from .backbones.vit_pytorch_uda import uda_vit_base_patch16_224_TransReID, uda_vit_small_patch16_224_TransReID
import torch.nn.functional as F
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import numpy as np
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 23, 3])
            print('using resnet101 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride,frozen_stages=cfg.MODEL.FROZEN)
            print('using se_resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but modelgot {}'.format(model_name))

        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading un_pretrain model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.LayerNorm(self.in_planes)
        
    def forward(self, x, label=None, cam_label=None, view_label=None, return_logits=False):  # label is unused if self.cos_layer == 'no'
        
        x = self.base(x, cam_label=cam_label)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if return_logits:
            cls_score = self.classifier(feat)
            return cls_score
        
        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        elif self.task_type == 'classify_DA': # test for classify domain adapatation
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score
        
        else:

            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            # if 'classifier' in i or 'arcface' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from revise {}'.format(trained_path))

    def load_un_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in self.state_dict():
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE
        if '384' in cfg.MODEL.Transformer_TYPE or 'small' in cfg.MODEL.Transformer_TYPE:
            self.in_planes = 384 
        else:
            self.in_planes = 384
        self.bottleneck_dim = 256
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.Transformer_TYPE))
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            if cfg.MODEL.Transformer_TYPE == 't2t_vit_14': # from FAZLI
                self.base = factory[cfg.MODEL.Transformer_TYPE](stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
            else:
                self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)


        #     self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_CROP, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        # else:
        #     self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self._load_parameter(pretrain_choice, model_path)

    def _load_parameter(self, pretrain_choice, model_path):
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading trans_tune model......from {}'.format(model_path))
        elif pretrain_choice == 'pretrain':
            self.load_param_finetune(model_path)
            print('Loading pretrained model......from {}'.format(model_path))

    def forward(self, x, label=None, cam_label= None, view_label=None, return_logits=False):  # label is unused if self.cos_layer == 'no'
        global_feat = self.base(x) #from FAZLI
        # global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)
        if return_logits:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score
        elif self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'bottleneck' in i or 'gap' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'module.' in i: new_i = i.replace('module.','') 
            else: new_i = i 
            if new_i not in self.state_dict().keys():
                print('model parameter: {} not match'.format(new_i))
                continue
            self.state_dict()[new_i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

class Discriminator(nn.Module):
    def __init__(self,in_dim, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(in_dim, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.l4 = nn.LogSoftmax(dim=1)
        self.slope = 0.2

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        x = self.l4(x)
        return x

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]

class build_uda_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_uda_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.task_type = cfg.MODEL.TASK_TYPE
        self.in_planes = 384 if 'small' in cfg.MODEL.Transformer_TYPE else 384
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.Transformer_TYPE))
        if cfg.MODEL.TASK_TYPE == 'classify_DA':
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_CROP, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, block_pattern=cfg.MODEL.BLOCK_PATTERN)
        else:
            self.base = factory[cfg.MODEL.Transformer_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, aie_xishu=cfg.MODEL.AIE_COE,local_feature=cfg.MODEL.LOCAL_F, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, use_cross=cfg.MODEL.USE_CROSS, use_attn=cfg.MODEL.USE_ATTN,  block_pattern=cfg.MODEL.BLOCK_PATTERN)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.discriminator = Discriminator(self.in_planes)
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))
        elif pretrain_choice == 'un_pretrain':
            self.base.load_un_param(model_path)
            print('Loading trans_tune model......from {}'.format(model_path))
        elif pretrain_choice == 'pretrain':
            if model_path == '':
                print('make model without initialization')
            else:
                self.load_param_finetune(model_path)
                print('Loading pretrained model......from {}'.format(model_path))

    def forward(self, x, x2, label=None, cam_label= None, view_label=None, domain_norm=False, return_logits=False, return_feat_prob=False, cls_embed_specific=False):  # label is unused if self.cos_layer == 'no'
        inference_flag = not self.training
        global_feat, global_feat2, global_feat3, cross_attn = self.base(x, x2, cam_label=cam_label, view_label=view_label, domain_norm=domain_norm, cls_embed_specific=cls_embed_specific, inference_target_only=inference_flag)

        
        if self.training:
            p_source = self.discriminator(global_feat)
            p_target = self.discriminator(global_feat2)

        if self.neck == '':
            feat = global_feat
            feat2 = global_feat2
            feat3 = global_feat3
        else:
            feat = self.bottleneck(global_feat) if self.training else None
            feat3 = self.bottleneck(global_feat3) if self.training else None
            feat2 = self.bottleneck(global_feat2)

        if return_logits:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
                cls_score2 = self.classifier(feat2, label)
                cls_score3 = self.classifier(feat3, label) if global_feat3 is not None else None
                
            else:
                cls_score = self.classifier(feat)   if global_feat is not None else None
                cls_score2 = self.classifier(feat2)
                cls_score3 = self.classifier(feat3) if global_feat3 is not None else None
                
                
            return cls_score, cls_score2, cls_score3

        if self.training or return_feat_prob:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label) if self.training else None
                cls_score2 = self.classifier(feat2, label)
                cls_score3 = self.classifier(feat3, label) if self.training else None
            else:
                cls_score = self.classifier(feat) if self.training else None
                cls_score2 = self.classifier(feat2)
                cls_score3 = self.classifier(feat3) if self.training else None
            if self.training:
                return (cls_score, global_feat, feat), (cls_score2, global_feat2, feat2), (cls_score3, global_feat3, feat3), cross_attn, p_source, p_target  # source , target , source_target_fusion
            else:
                return (cls_score, global_feat, feat), (cls_score2, global_feat2, feat2), (cls_score3, global_feat3, feat3), cross_attn
        
            
        else:
            if self.neck_feat == 'after' and self.neck != '':
                # print("Test with feature after BN")
                return feat, feat2, feat3
                
            else:
                # print("Test with feature before BN")
                return global_feat, global_feat2, global_feat3  # source , target , source_target_fusion
                

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i or 'bottleneck' in i or 'gap' in i:
                continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))


    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:  
            if 'module.' in i: new_i = i.replace('module.','') 
            else: new_i = i 
            if new_i not in self.state_dict().keys():
                print('model parameter: {} not match'.format(new_i))
                continue
            self.state_dict()[new_i].copy_(param_dict[i])

        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_hh = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID, 
    'cvt_21_224_TransReID': cvt_21_224_TransReID,
    'uda_cvt_21_224_TransReID': uda_cvt_21_224_TransReID,
    # 't2t_vit_14': t2t_vit_14,
    'uda_vit_small_patch16_224_TransReID': uda_vit_small_patch16_224_TransReID, 
    'uda_vit_base_patch16_224_TransReID': uda_vit_base_patch16_224_TransReID,

    # 'resnet101': resnet101,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.BLOCK_PATTERN == '3_branches':
            model = build_uda_transformer(num_class, camera_num, view_num, cfg, __factory_hh)
            print('===========building uda transformer===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_hh)
            print('===========building transformer===========')
    else:
        print('===========ResNet===========')
        model = Backbone(num_class, cfg)
    return model
