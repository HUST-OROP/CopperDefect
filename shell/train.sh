time=$(date "+%Y%m%d")

work_folder=./work_dir/${time}/

for model in frcn_hrnet_w40;
do
python ./tools/train.py \
    ./configs/depth/${model}_depth.py \
    --work-dir ${work_folder}/${model} \
    --gpu-ids 0 \
    --cfg-options "model.rpn_head.anchor_generator.scales=[2,4,8]"
done
