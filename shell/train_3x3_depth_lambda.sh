time=$(date "+%Y%m%d")

work_folder=./Work_dir/${time}/depth/lambda/

split=3x3

model=frcn_hrnet_w40

for weight in  0.05 0.1 0.2 0.5 1 2 5;
do
python ./mmdetection/tools/train.py \
    ./Model/full_data/depth/${split}/${model}_depth.py \
    --work-dir ${work_folder}/anchor_248/${split}/${model}/${weight} \
    --gpu-ids 1 \
    --cfg-options "model.rpn_head.anchor_generator.scales=[2,4,8]" "model.roi_head.bbox_head.loss_depth.loss_weight=${weight}"
done

for weight in 0.05 0.1 0.2 0.5 1 2 5;
do
python ./mmdetection/tools/train.py \
    ./Model/full_data/depth/${split}/${model}_depth.py \
    --work-dir ${work_folder}/anchor_48/${split}/${model}/${weight} \
    --gpu-ids 1 \
    --cfg-options "model.rpn_head.anchor_generator.scales=[2,4,8]" "model.roi_head.bbox_head.loss_depth.loss_weight=${weight}"
done