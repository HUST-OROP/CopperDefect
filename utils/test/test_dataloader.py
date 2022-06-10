#%%
#### check dataloader from config file
from mmcv import Config
from mmcv.utils import config
from mmdet.datasets import build_dataset,build_dataloader
import time 

timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

config_file = "/home/user/sun_chen/Projects/ZJDetection/Model/full_data/frcn_depth.py"

# base_config = Config.fromfile(base_config_path)
cfg = Config.fromfile(config_file)
# for cfg in [base_config,novel_config]:
datasets = build_dataset(
        cfg.data.train,
        )

data_loaders =  build_dataloader(
            datasets,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            seed=0,
            runner_type="EpochBasedRunner",
            persistent_workers=cfg.data.get('persistent_workers', False))
# print(datasets)
for i, data_batch in enumerate(data_loaders):
    print(data_batch["gt_depth"])
    print(data_batch["gt_labels"])
# %%
