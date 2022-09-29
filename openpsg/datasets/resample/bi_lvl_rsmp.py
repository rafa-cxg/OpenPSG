import json
import os
from collections import OrderedDict
from typing import Dict
import torch
import numpy as np
import pickle


from ..builder import  RESAMPLE
@RESAMPLE.register_module()
class BGNN_Resample():
    def __init__(
            self,
            method,
            RESAMPLING_PARAM,
            path='',
            drop_rate=0.4

    ):
        self.method=method
        self.path=path
        self.drop_rate=drop_rate
        self.RESAMPLING_PARAM_REPEAT_FACTOR=RESAMPLING_PARAM.REPEAT_FACTOR



    def __call__(self,index: int, relation: np.ndarray,
                         repeat_dict: Dict, drop_rate): #apply_resampling():
        """
        这个函数完全是操作的instancesampling而不是图片级别
        Args:
            index:
            relation: N x 3 array
            repeat_dict: r_c, rc_cls image repeat number and repeat number of each category
            drop_rate:

        Returns:

        """
        relation_non_masked = relation.copy()

        # randomly drop the head and body categories for more balance distribution
        # reduce duplicate head and body for more balances
        rc_cls = repeat_dict['cls_rf']#dict of relation repeat times，所有图片都一样的
        r_c = repeat_dict[index]

        if r_c > 1:
            # rc <= 1,
            # no need repeat this images, just return

            selected_rel_idx = []
            for i, each_rel in enumerate(relation):#relation是一张图片的
                rel_label = each_rel[-1]

                if rc_cls.get(rel_label) is not None:
                    selected_rel_idx.append(i)

            # decrease the head classes of repeated image and non-repeated image
            # if the images are repeated, the total times are > 1, then we calculate the decrease time
            # according to the repeat times.
            # head_drop_rate is reduce the head instance num from the initial.

            if len(selected_rel_idx) > 0:
                selected_head_rel_idx = np.array(selected_rel_idx, dtype=int)
                ignored_rel = np.random.uniform(0, 1, len(selected_head_rel_idx))
                total_repeat_times = r_c#当前图片重复次数


                rel_repeat_time = np.array([rc_cls[rel] for rel in relation[:, -1]])#生成该图片所有rel的重复次数

                drop_rate = (1 - (rel_repeat_time / (total_repeat_times + 1e-11) ))  * drop_rate#该rel需重复次数越多，drop rate越小，但该图片出现次数越多，drop越大
                # print((1 - (rel_repeat_time / (total_repeat_times + 1e-11))) * 1.5)
                # print(drop_rate)
                ignored_rel = ignored_rel < np.clip(drop_rate, 0.0, 1.0)
                # ignored_rel[rel_repeat_time >= 2] = False

                # if len(np.nonzero(ignored_rel == 0)[0]) == 0:
                #     ignored_rel[np.random.randint(0, len(ignored_rel))] = False
                selected_head_rel_idx = np.array(selected_head_rel_idx, dtype=int)
                relation[selected_head_rel_idx[ignored_rel], -1] = -1#要抛弃的rel被标为-1


        return relation, relation_non_masked
    def resampling_dict_generation(self,dataset, category_list):

        # logger.info("using resampling method:" + dataset.resampling_method)
        repeat_dict_dir = self.path
        curr_dir_repeat_dict = os.path.join(repeat_dict_dir, "repeat_dict.pkl")
        if repeat_dict_dir is not None and repeat_dict_dir != "" or os.path.exists(curr_dir_repeat_dict):
            if os.path.exists(curr_dir_repeat_dict):
                repeat_dict_dir = curr_dir_repeat_dict

            # logger.info("load repeat_dict from " + repeat_dict_dir)
            with open(repeat_dict_dir, 'rb') as f:
                repeat_dict = pickle.load(f)

            return repeat_dict

        else:
            # logger.info(
            #     "generate the repeat dict according to hyper_param on the fly")

            if self.method in ["bilvl", 'lvis']:
                # when we use the lvis sampling method,
                global_rf = self.RESAMPLING_PARAM_REPEAT_FACTOR
                # logger.info(f"global repeat factor: {global_rf};  ")
                if self.method == "bilvl":
                    # share drop rate in lvis sampling method
                    self.drop_rate =  self.drop_rate
                    # logger.info(f"drop rate: {dataset.drop_rate};")
                else:
                    self.drop_rate = 0.0#todo 0有什么意义？
            else:
                raise NotImplementedError(self.method)

            F_c = np.zeros(len(category_list)+1)#算上和背景类
            for i in range(len(dataset)):
                anno = dataset[i]
                tgt_rel_matrix = torch.tensor(anno['relations'])
                # tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
                # assert tgt_pair_idxs.shape[1] == 2
                # tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
                # tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
                # tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs,
                #                               tgt_tail_idxs].contiguous().view(-1)
                tgt_rel_labs=tgt_rel_matrix[:,2]
                for each_rel in tgt_rel_labs:
                    F_c[each_rel] += 1

            total = sum(F_c)
            F_c /= (total + 1e-11)

            rc_cls = {
                i: 1 for i in range(len(category_list))
            }
            global_rf = self.RESAMPLING_PARAM_REPEAT_FACTOR

            reverse_fc = global_rf / (F_c[1:] + 1e-11)
            reverse_fc = np.sqrt(reverse_fc)
            final_r_c = np.clip(reverse_fc, a_min=1.0, a_max=np.max(reverse_fc) + 1)
            # quantitize by random number
            rands = np.random.rand(*final_r_c.shape)
            _int_part = final_r_c.astype(int)
            _frac_part = final_r_c - _int_part
            rep_factors = _int_part + (rands < _frac_part).astype(int)

            for i, rc in enumerate(rep_factors.tolist()):
                rc_cls[i + 1] = int(rc)

            repeat_dict = {}
            for i in range(len(dataset)):
                anno = dataset[i]
                tgt_rel_matrix = torch.tensor(anno['relations'])
                tgt_rel_labs=tgt_rel_matrix[:,2].numpy()
                hit_rel_labels_r_c = []
                curr_rel_lables = []

                for rel_label in tgt_rel_labs:
                    if rel_label not in curr_rel_lables:
                        curr_rel_lables.append(rel_label)
                        hit_rel_labels_r_c.append(rc_cls[rel_label])

                hit_rel_labels_r_c = np.array(hit_rel_labels_r_c)

                r_c = 1
                if len(hit_rel_labels_r_c) > 0:
                    r_c = int(np.max(hit_rel_labels_r_c))
                repeat_dict[i] = r_c

            repeat_dict['cls_rf'] = rc_cls


            return repeat_dict