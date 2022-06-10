# time=$(date "+%Y%m%d")
time=20220306
work_folder=./Work_dir/${time}/depth/

split=3x3

# for model in frcn frcn_r101 frcn_hrnet_w32 frcn_hrnet_w40;
# do
# python ./mmdetection/tools/train.py \
#     ./Model/full_data/depth/${split}/${model}_depth.py \
#     --work-dir ${work_folder}/anchor_248/${split}/${model} \
#     --gpu-ids 1 \
#     --cfg-options "model.rpn_head.anchor_generator.scales=[2,4,8]"
# done

# for model in frcn frcn_r101 frcn_hrnet_w32 frcn_hrnet_w40;
# do
# python ./mmdetection/tools/train.py \
#     ./Model/full_data/depth/${split}/${model}_depth.py \
#     --work-dir ${work_folder}/anchor_48/${split}/${model} \
#     --gpu-ids 1 \
#     --cfg-options "model.rpn_head.anchor_generator.scales=[4,8]"
# done

for model in frcn frcn_r101 frcn_hrnet_w32 frcn_hrnet_w40;
do
python ./mmdetection/tools/test.py \
    ${work_folder}/anchor_48/${split}/${model}/${model}_depth.py \
    ${work_folder}/anchor_48/${split}/${model}/best_mAP_epoch.pth \
    --format-only \
    --options "jsonfile_prefix=./${model}_results"
done
