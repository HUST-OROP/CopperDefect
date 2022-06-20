import mmcv
import os
import torch
from torch.distributed import launch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel,MMDistributedDataParallel
from mmcv.runner import (get_dist_info, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import setup_multi_processes
from environment import set_envir
set_envir()
from copperdet import *

from mmdet.datasets import build_dataloader,build_dataset
from analysis_results import ResultVisualizer,bbox_map_eval
from mmdet.datasets import get_loading_pipeline

def model_output(cfg,ckpt_path,gpu_id=[0]):

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # build the model and load checkpoint
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.pop('frozen_parameters', None)

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt_path, map_location='cpu')
    model = fuse_conv_bn(model)
    # in case the test dataset is concatenated
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    

    cfg.gpu_ids = gpu_id
    rank, _ = get_dist_info()

    model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()


    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric="mAP"))
    metric = dataset.evaluate(results, **eval_kwargs)
    print(metric)

    return results

def bbox_visualize(cfg,outputs,show_dir,gt_only=False):
            
    os.makedirs(show_dir,exist_ok=True)
    cfg.data.test.pop('samples_per_gpu', 0)
    if cfg.data.train.type == "NWayKShotDataset":
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.multi_pipelines.query)
    elif cfg.data.train.type == "TwoBranchDataset":
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.dataset.multi_pipelines.main)      
    else:
        cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline) 
    cfg.data.test.filter_empty_gt=True
    result_visualizer = ResultVisualizer(score_thr=0.3)
    if gt_only:
        datasets = build_dataset(cfg.data.test)
        result_visualizer._save_img_gt(datasets,show_dir)
    else:
        cfg.data.test.test_mode = True
        datasets = build_dataset(cfg.data.test)
        _mAPs = {}
        prog_bar = mmcv.ProgressBar(len(outputs))
        for i, (result, ) in enumerate(zip(outputs)):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = datasets.prepare_train_img(i)
            mAP = bbox_map_eval(result, data_info['ann_info'])
            _mAPs[i] = mAP
            # _mAPs[i] = 1
            prog_bar.update()

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        result_visualizer._save_image_gts_results(datasets, outputs, _mAPs, show_dir)

def save2json(cfg,results,save_path=None):
    if "data" in cfg.keys():
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
    else:
        cfg.test_mode = True
        dataset = build_dataset(cfg)
    json_file = dataset._det2json(results)
    if save_path!=None:
        mmcv.dump(json_file,save_path)
    return json_file

if __name__=="__main__":

    show_dir = "/home/sunchen/Projects/CopperDetetion/dataset/train_vis"
    cfg_path = "/home/sunchen/Projects/CopperDetetion/configs/depth/frcn_hrnet_w40_detection.py"

    cfg = Config.fromfile(cfg_path)
    
    ckpt_path = "/home/sunchen/Projects/CopperDetetion/work_dir/20220618/frcn_hrnet_w40/best_mAP_epoch_6.pth"
    # result_file = os.path.join(show_dir,"results.pkl")
    result_file = "/home/sunchen/Projects/CopperDetetion/nms_result.pkl"
    # result_file = "/home/sunchen/Projects/CopperDetetion/results.pkl"
    # outputs = model_output(cfg,ckpt_path,gpu_id=[0])

    # mmcv.dump(outputs, result_file)
    outputs = mmcv.load(result_file)
    
    cfg.data.test.use_slice=False
    bbox_visualize(cfg,outputs=outputs,show_dir=show_dir)


    # for method in dataset_list:
    #     file_list = [os.path.join(path,method,"frcn/640^2",file_name) for file_name in os.listdir(os.path.join(path,method,"frcn/640^2"))]
        
    #     for file_name in file_list:
    #         if os.path.splitext(file_name)[-1] == ".py":
    #             cfg_path=os.path.join(path,method,"frcn/640^2",file_name)
                
    #     for file_name in file_list:   
    #         if "epoch_24" in file_name:
    #             ckpt_path=os.path.join(path,method,"frcn/640^2",file_name)

    #     result_file = show_dir+f"ckpt/{method}.pkl"

    #     cfg = Config.fromfile(cfg_path)
    #     # cfg.data.test = cfg.data.val
    #     # cfg.merge_from_dict("")
    #     cfg.evaluation.iou_thrs=[0.5]
    #     outputs = model_output(cfg,ckpt_path,gpu_id=[0])
        # mmcv.dump(outputs, result_file)
        
        # outputs = mmcv.load(result_file)
        # bbox_visualize(cfg,outputs,show_dir+f"/visualization/det/{method}")
        
        # from confusion_matrix import calculate_confusion_matrix,plot_confusion_matrix,main
        # import argparse
        # args = argparse.Namespace(
        #     config = cfg_path,
        #     cfg_options = None,
        #     prediction_path = result_file,
        #     nms_iou_thr = None,
        #     show =False,
        #     score_thr = 0,
        #     tp_iou_thr = 0.5,
        #     save_dir = f"/home/user/sun_chen/Projects/KDTFA/results/confusion_matrix/"
        #     )    
        # main(args)

        # if isinstance(cfg.data.test, dict):
        #     cfg.data.test.test_mode = True
        # elif isinstance(cfg.data.test, list):
        #     for ds_cfg in cfg.data.test:
        #         ds_cfg.test_mode = True
        # dataset = build_dataset(cfg.data.test)

        # confusion_matrix = calculate_confusion_matrix(dataset, outputs,
        #                                             0,
        #                                             None,
        #                                             0.5)
        # plot_confusion_matrix(
        #     confusion_matrix,
        #     dataset.CLASSES + ('background', ),
        #     save_dir=args.save_dir,
        #     show=args.show)