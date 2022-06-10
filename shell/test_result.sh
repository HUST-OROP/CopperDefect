
# time=20220107
# # $(date "+%Y%m%d")
# type=detection #depth

# work_folder=./Work_dir/${time}/${type}/
# split=3x3
# # model=cascade
# # for split in 2x2 3x3;
# # do 
# for model in cascade hrnet frcn frcn_copy_paste retina;
# do
# echo #########################
# echo ${work_folder}/${split}/${model}
# echo #########################

python ./mmdetection/tools/test.py \
    ./Model/full_data/${type}/${split}/${model}.py \
    ${work_folder}/${split}/${model}/epoch_24.pth \
    --out ${work_folder}/${split}/${model}/results.pkl

# python ./mmdetection/tools/analysis_tools/analyze_results.py \
#     ./Model/full_data/${type}/${split}/${model}.py \
#     ${work_folder}/${split}/${model}/results.pkl \
#     ./Result/Visual/${type}/${split}/${model}    

# done
# done


######## visual ########

# python ./mmdetection/tools/test.py \
# /home/user/sun_chen/Projects/ZJDetection/Work_dir/20220306/depth/anchor_48/3x3/frcn_hrnet_w40/frcn_hrnet_w40_depth.py \
# /home/user/sun_chen/Projects/ZJDetection/Work_dir/20220306/depth/anchor_48/3x3/frcn_hrnet_w40/best_mAP_epoch_23.pth \
# --out /home/user/sun_chen/Projects/ZJDetection/Result/frcn_hrnet_w40_3x3.pkl

# python /home/user/sun_chen/Projects/ZJDetection/mmdetection/tools/analysis_tools/coco_error_analysis.py \
# /home/user/sun_chen/Projects/ZJDetection/Work_dir/20220306/depth/anchor_48/3x3/frcn_hrnet_w40/frcn_hrnet_w40_depth.py \
# /home/user/sun_chen/Projects/ZJDetection/Result/frcn_hrnet_w40_3x3.pkl \
# ./Result/Visual/frcn_hrnet_w40_3x3 \
# --topk 1000
