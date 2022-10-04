# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import ipdb
import torch
import math
from ..relation_heads.approaches.motif_util import \
    obj_edge_vectors
from mmdet.core import bbox_overlaps
from torch import nn
from torch.nn import functional as F,init



class TwoStagePredictor(nn.Module):
    def __init__(self,  cfg,object_classes,num_obj_cls,num_rel_cls):
        super(TwoStagePredictor, self).__init__()
        self.cfg=cfg
        self.num_obj_cls = num_obj_cls
        self.num_rel_cls = num_rel_cls
        self.object_classes=object_classes
        self.word_dim=200
        self.geometry_feat_dim = 128
        self.input_dim = 1024+1024+1024
        self.hidden_dim = 4096
        #待定self.context_layer=
        # post classification
        obj_embed_vecs = obj_edge_vectors( self.object_classes+['background'], wv_dir=self.cfg.glove_dir,
                                          wv_dim=self.word_dim)
        self.sub_embed = nn.Embedding(len(self.object_classes) + 1, self.word_dim)
        self.obj_embed = nn.Embedding(len(self.object_classes) + 1, self.word_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.sub_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
        self.sub_proj = nn.Sequential(
            *[
                make_fc(200, 1024),
                nn.ReLU(inplace=True),
            ]
        )
        self.obj_proj=nn.Sequential(
            *[
                make_fc(200,1024),
                nn.ReLU(inplace=True),
            ]
        )
        self.interact_embed = nn.Sequential(
            *[
                make_fc(9*2+2, 1024),
                nn.ReLU(inplace=True),
            ]
        )
        self.rel_classifier1 = build_classifier(self.input_dim, self.hidden_dim)
        self.rel_classifier2 = build_classifier(self.hidden_dim, self.num_rel_cls+1)
        # self.confidence_score = build_classifier(1024, 1)
        self.LN1=torch.nn.LayerNorm([self.hidden_dim])
        self.LN2 = torch.nn.LayerNorm(51)
        self.sigmoid = nn.Sigmoid()
        self.softmax =nn.Softmax(dim=-1)
        self.relu=nn.LeakyReLU(0.2)
        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier1.reset_parameters()
        self.rel_classifier2.reset_parameters()
    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()
    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)
    def forward(
        self,
        image_metas,
        sub_boxs,
        obj_boxs,
        rel_pair_idxs,
    ):
        # sub_boxs=sub_boxs.detach()
        # obj_boxs=obj_boxs.detach()
        rel_infos = (encode_rel_box_info(image_metas, sub_boxs, obj_boxs, rel_pair_idxs))
        rel_infos=torch.cat(rel_infos,0)
        pos_embed=[]

        # for sub_box, obj_box, rel_pair_idx,image_meta,rel_info in zip(sub_boxs[:], obj_boxs[:] ,rel_pair_idxs[:],image_metas,rel_infos):
        sub_prop_infos = encode_box_info(sub_boxs)  # todo 重复计算会有点慢，改为一次计算不重复的所有box
        obj_prop_infos = encode_box_info(obj_boxs)
        sub_prop_infos=torch.cat(sub_prop_infos,0)
        obj_prop_infos = torch.cat(obj_prop_infos, 0)
        sub_proj = self.sub_proj(self.sub_embed(rel_pair_idxs[:,:,0]))
        obj_proj= self.sub_proj(self.obj_embed(rel_pair_idxs[:,:,1]))
        pos_embed.append(torch.cat((sub_prop_infos,obj_prop_infos,rel_infos), -1))
        pos_embed = self.interact_embed(torch.cat(pos_embed, 0))
        input=torch.cat((sub_proj, obj_proj, pos_embed), -1)

        rel_feats = self.rel_classifier1(input)#[N,4]
        # rel_feats = self.LN1(rel_feats)
        rel_feats = self.relu(rel_feats)
        rel_cls_logits = self.rel_classifier2(rel_feats)  # [N,4]
        # rel_cls_logits = self.LN2(rel_cls_logits)
        rel_cls_logits=(rel_cls_logits)
        # rel_cls_logits=confidence_score*rel_cls_logits
        rel_cls_logits = self.relu(rel_cls_logits)





        return  rel_cls_logits

from mmdet.core import bbox_cxcywh_to_xyxy
def encode_box_info(boxs):
    boxes_info = []
    for box  in (boxs):

        w, h = box[:,2],box[:,3]
        x, y = box[:,0],box[:,1]
        box = bbox_cxcywh_to_xyxy(box)
        x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]

        info = torch.stack([
            w , h , x , y , x1 , y1 , x2 ,
            y2, w * h
        ],
            dim=-1).view(-1, 9)
        boxes_info.append(info.unsqueeze(0))

    return boxes_info # torch.cat(boxes_info, dim=0)
def encode_rel_box_info(meta,sub_boxs,obj_boxs,rel_pair_idxs):#todo 和上面的函数有很多重叠运算需要优化
    """

    """


    sizes=[meta[i]['img_shape'] for i in range(len(meta))]

    boxes_info = []
    for sub_box,obj_box,rel_pair_idx,siz in zip(sub_boxs,obj_boxs,rel_pair_idxs,sizes):
        img_size = siz
        wid = img_size[0]
        hei = img_size[1]
        assert wid * hei != 0
        sub_xy = sub_box[:, :2]
        sub_x, sub_y = sub_xy.split([1,1], dim=-1)
        sub_area=sub_box[:, 2]*sub_box[:, 3]
        # sub_xy = sub_box[:, :2] + 0.5 * sub_wh
        # sub_wh = sub_box[:, 2:] - sub_box[:, :2] + 1.0

        '''obj_box'''
        # obj_wh = obj_box[:, 2:] - obj_box[:, :2] + 1.0
        obj_area = obj_box[:, 2] * obj_box[:, 3]
        obj_xy = obj_box[:, :2]
        obj_x, obj_y = obj_xy.split([1, 1], dim=-1)
        # obj_w, obj_h = obj_wh.split([1, 1], dim=-1)
        # obj_x, obj_y = obj_xy.split([1, 1], dim=-1)#
        # obj_x1, obj_y1, obj_x2, obj_y2 = obj_box.split([1, 1, 1, 1], dim=-1)

        distance=torch.pow(( torch.pow((sub_x-obj_x),2)+torch.pow((sub_y-obj_y),2)),0.5)

        # iou_=(sub_x2-obj_x1)*(sub_y2-obj_y1)/(obj_h*obj_w+sub_h*sub_w-(sub_x2-obj_x1)*(sub_y2-obj_y1))#错的
        sub_box = bbox_cxcywh_to_xyxy(sub_box)
        obj_box = bbox_cxcywh_to_xyxy(obj_box)
        iou=(bbox_overlaps(sub_box, obj_box, is_aligned=True,mode='iou')).unsqueeze(-1)
        iou[iou<0]=0#+1e-12
        distance=distance#+1e-12#todo 我该不该只加等于0的？
        info =torch.cat((iou,distance), dim=1).unsqueeze(0)# (iou - distance)/10
        # info=torch.cat((iou,distance), dim=1).view(-1, 2)
        # info=torch.nn.functional.normalize(info, p=2.0, dim=0, eps=1e-12, out=None)

        boxes_info.append(info)

    return boxes_info#torch.cat(boxes_info, dim=0)
def boxlist_iou(sub_box,area1, obj_box,area2):
#必须是xyxy形式

    N = len(sub_box)
    M = len(obj_box)



    box1 = sub_box
    box2 = obj_box

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
def make_fc(dim_in, hidden_dim, use_gn=False):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''

    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc
class DotProductClassifier(nn.Module):
    def __init__(self, in_dims, num_class, bias=True, learnable_scale=False):
        super(DotProductClassifier, self).__init__()
        self.in_dims = in_dims
        self.weight = nn.Parameter(torch.Tensor(num_class, in_dims))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_class))
        self.scales = None
        if learnable_scale:
            self.scales = nn.Parameter(torch.ones(num_class))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def fix_weights(self, requires_grad=False):
        self.weight.requires_grad = requires_grad
        if self.bias is not None:
            self.bias.requires_grad = requires_grad

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.scales is not None:
            output *= self.scales

        return output
def build_classifier(input_dim, num_class, bias=True):
    return DotProductClassifier(input_dim, num_class, bias,#4096
                                    False)