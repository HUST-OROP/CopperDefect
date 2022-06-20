import json
import os
import os.path as osp
from collections import OrderedDict
from typing import List

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from sahi.slicing import slice_coco
from terminaltables import AsciiTable

from ..bbox.mean_ap import eval_map_depth
from mmdet.core.post_processing import multiclass_nms


def save_json(json_path,json_dict):
    with open(json_path,"w") as fp:
        json.dump(json_dict,fp,indent=4,separators=(",",": "))

@DATASETS.register_module()
class ZJDataset(CocoDataset):
    CLASSES = ("lizi",)
    
    def __init__(self, 
                 ann_file,
                 img_prefix,
                 data_root=None,
                 sliced_image_folder:str=None,
                 sliced_anno_path:str=None,
                 slice_size:List[int]=[1024,1024],
                 min_area_ratio:float=0.1,
                 overlap_ratio:List[float]=[0.1,0.1],
                 slice_process=False,
                 use_slice=False,
                 **kwargs):
        self.raw_annofile = ann_file
        self.raw_id2name={}
        
        if slice_process and not os.path.exists(sliced_anno_path):
            assert sliced_image_folder is not None and sliced_anno_path is not None
            
            if data_root is not None:
                if not osp.isabs(ann_file):
                    ori_anno_path = osp.join(data_root,ann_file)
                else:
                    ori_anno_path = ann_file
                if not (img_prefix is None or osp.isabs(img_prefix)):
                    ori_img_folder = osp.join(data_root, img_prefix)
                else:
                    ori_img_folder = img_prefix
            else:
                ori_anno_path = ann_file
                ori_img_folder = img_prefix
                
            coco_dict, _ = slice_coco(
                coco_annotation_file_path=ori_anno_path,
                image_dir=ori_img_folder,
                output_coco_annotation_file_name=None,
                output_dir=sliced_image_folder,
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                min_area_ratio=min_area_ratio,
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1],
            )
            for i in coco_dict["annotations"]:
                if i["iscrowd"] != 0:
                    i["depth"]=i["iscrowd"]
                    i["iscrowd"]=0
                    
            save_json(sliced_anno_path,coco_dict)
            print(f"Sliced dataset for 'slice_size: {slice_size}' is exported to {sliced_image_folder}")
        if use_slice:
            assert os.path.exists(sliced_anno_path) and not len(os.listdir(sliced_image_folder))==0
            ann_file = sliced_anno_path
            img_prefix = sliced_image_folder
        super(ZJDataset, self).__init__(ann_file=ann_file,img_prefix=img_prefix,**kwargs)

    def get_raw_anno_info(self):
        cocodata = COCO(self.raw_annofile)
        self.raw_data = cocodata
        anno_info = []
        img_ids = cocodata.get_img_ids()
        for id in img_ids:
            ann_ids = cocodata.get_ann_ids(img_ids=[id])
            ann_info = cocodata.load_anns(ann_ids)
            data_info = cocodata.load_imgs([id])[0]
            data_info['filename'] = data_info['file_name']
            anno_info.append(self._parse_ann_info(data_info, ann_info))
            self.raw_id2name[id]=data_info['file_name']
        return anno_info

    def postprocess(self,results,
                    score_thr:float=0.05,
                    postproscess_type:str="nms",
                    iou_threshold:float=0.5,
                    **kwargs):
        
        raw_annoinfo = self.get_raw_anno_info()
        raw_name_list = [ann["seg_map"][:-4] for ann in raw_annoinfo]
        remap_list = {k:[] for k in raw_name_list}
        json_data = self._det2json(results)
        
        for anno_data in json_data:
            img_name = anno_data["image_name"]
            raw_name,x_1,y_1,x_2,y_2 = img_name.split("_")
            x_1 = float(x_1)
            y_1 = float(y_1)
            nms_box = [anno_data["bbox"][0]+x_1,anno_data["bbox"][1]+y_1,
                    anno_data["bbox"][2]+anno_data["bbox"][0]+x_1,anno_data["bbox"][3]+anno_data["bbox"][1]+y_1,
                    anno_data["score"],1-anno_data["score"]]
            remap_list[raw_name].append(nms_box)
            
        processed_results = []
        for k, v in remap_list.items():
            v = np.array(v,dtype=np.float32)
            if len(v.shape)==2:
                bboxes = torch.from_numpy(v[:,:4])
                scores = torch.from_numpy(v[:,4:6])
                selected_boxes,selected_labels = multiclass_nms(bboxes,scores,score_thr=score_thr,\
                                nms_cfg=dict(type=postproscess_type,iou_threshold=iou_threshold,**kwargs))
                nms_result = selected_boxes.numpy()
            else:
                nms_result = np.zeros((0, 5), dtype=np.float32)
            processed_results.append([nms_result])
    
    def single_evaluate(self,
                 results,
                 annotations,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        
        return eval_results 

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 standard_eval=True,
                 postprocess=False,
                 postprocess_cfg=None,
                 scale_ranges=None):

        if postprocess and postprocess_cfg is not None:
            raw_results = self.postprocess(results,postprocess_cfg)
            raw_annotations = self.get_raw_anno_info()
            raw_eval_results = self.single_evaluate(raw_results,
                                                    raw_annotations,
                                                    metric=metric,
                                                    logger=logger,
                                                    proposal_nums=proposal_nums,
                                                    iou_thr=iou_thr,
                                                    scale_ranges=scale_ranges)      
        
        if standard_eval:
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            standard_eval_results = self.single_evaluate(results,
                                                         annotations,
                                                         metric=metric,
                                                         logger=logger,
                                                         proposal_nums=proposal_nums,
                                                         iou_thr=iou_thr,
                                                         scale_ranges=scale_ranges)
            
@DATASETS.register_module()
class ZJDepthDataset(ZJDataset):
    def __init__(self, **kwargs):
        super(ZJDepthDataset, self).__init__(**kwargs)
        
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_depth = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_depth.append(ann.get("depth",None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_depth = np.array(gt_depth,dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_depth = np.array(0,dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            depth=gt_depth)

        return ann
    
    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        id2name = {i["id"]:i["file_name"] for i in self.data_infos}
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data["image_name"] = id2name[img_id]
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data["depth"]= bboxes[i][-1]
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results
            
    def postprocess(self,results,
                score_thr:float=0.05,
                postproscess_type:str="nms",
                iou_threshold:float=0.5,
                **kwargs):
        
        raw_annoinfo = self.get_raw_anno_info()
        raw_name_list = [ann["seg_map"][:-4] for ann in raw_annoinfo]
        remap_list = {k:[] for k in raw_name_list}
        json_data = self._det2json(results)
        
        for anno_data in json_data:
            img_name = anno_data["image_name"]
            raw_name,x_1,y_1,x_2,y_2 = img_name.split("_")
            x_1 = float(x_1)
            y_1 = float(y_1)
            nms_box = [anno_data["bbox"][0]+x_1,anno_data["bbox"][1]+y_1,
                    anno_data["bbox"][2]+anno_data["bbox"][0]+x_1,anno_data["bbox"][3]+anno_data["bbox"][1]+y_1,
                    anno_data["score"],1-anno_data["score"],anno_data["depth"]]
            remap_list[raw_name].append(nms_box)
            
        processed_results = []
        for k, v in remap_list.items():
            v = np.array(v,dtype=np.float32)
            if len(v.shape)==2:
                bboxes = torch.from_numpy(v[:,:4])
                scores = torch.from_numpy(v[:,4:6])
                heights = torch.from_numpy(v[:,-1]).reshape(-1,1)
                # selected_boxes,selected_labels,selected_indx = multiclass_nms(bboxes,scores,score_thr=0.05,
                #                 nms_cfg=dict(type='nms', iou_threshold=0.5),
                #                 return_inds=True)
                selected_boxes,selected_labels,selected_indx = multiclass_nms(bboxes,scores,score_thr=score_thr,\
                                nms_cfg=dict(type=postproscess_type,iou_threshold=iou_threshold,**kwargs),
                                return_inds=True)

                selected_heights = heights[selected_indx]
                nms_result = torch.concat((selected_boxes,selected_heights),dim=1).numpy()
            else:
                nms_result = np.zeros((0, 6), dtype=np.float32)
            processed_results.append([nms_result])
        return processed_results
    
    def single_evaluate(self,
                 results,
                 annotations,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate in VOC protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'mAP', 'recall'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple], optional): Scale ranges for evaluating
                mAP. If not specified, all bounding boxes would be included in
                evaluation. Default: None.

        Returns:
            dict[str, float]: AP/recall metrics.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, final_results = eval_map_depth(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=self.CLASSES,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                
                recall = 0
                precision = 0
                f1 = 0
                
                if len(final_results[0]["recall"])>0:
                    recall = final_results[0]["recall"][-1]
                    precision = final_results[0]["precision"][-1]
                    f1 = 2*recall*precision/(recall+precision+1e-6)
                # rmse = 
                eval_results[f'Precision'] = round(precision, 3)
                eval_results[f'Recall'] = round(recall, 3)
                eval_results[f'F1'] = round(f1, 3)
                eval_results[f'RMSE'] = round(final_results[0]["depth_rmse"], 3)
                            
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                proposal_nums,
                iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(proposal_nums):
                for j, iou_thr in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        
        return eval_results

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 standard_eval=True,
                 postprocess=False,
                 postprocess_cfg=None,
                 scale_ranges=None):

        if postprocess and postprocess_cfg is not None:
            raw_results = self.postprocess(results,**postprocess_cfg)
            raw_annotations = self.get_raw_anno_info()
            raw_eval_results = self.single_evaluate(results=raw_results,
                                                    annotations=raw_annotations,
                                                    metric=metric,
                                                    logger=logger,
                                                    proposal_nums=proposal_nums,
                                                    iou_thr=iou_thr,
                                                    scale_ranges=scale_ranges)      

        if standard_eval:
            annotations = [self.get_ann_info(i) for i in range(len(self))]
            standard_eval_results = self.single_evaluate(results=results,
                                                         annotations=annotations,
                                                         metric=metric,
                                                         logger=logger,
                                                         proposal_nums=proposal_nums,
                                                         iou_thr=iou_thr,
                                                         scale_ranges=scale_ranges)