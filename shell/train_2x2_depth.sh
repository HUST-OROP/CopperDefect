time=$(date "+%Y%m%d")

work_folder=./Work_dir/${time}/depth/

split=2x2

for model in frcn frcn_r101 frcn_hrnet_w32 frcn_hrnet_w40;
do
python ./mmdetection/tools/train.py \
    ./Model/full_data/depth/${split}/${model}_depth.py \
    --work-dir ${work_folder}/anchor_248/${split}/${model} \
    --gpu-ids 0 \
    --cfg-options "model.rpn_head.anchor_generator.scales=[2,4,8]"
done

for model in frcn frcn_r101 frcn_hrnet_w32 frcn_hrnet_w40;
do
python ./mmdetection/tools/train.py \
    ./Model/full_data/depth/${split}/${model}_depth.py \
    --work-dir ${work_folder}/anchor_48/${split}/${model} \
    --gpu-ids 0 \
    --cfg-options "model.rpn_head.anchor_generator.scales=[4,8]"
done
