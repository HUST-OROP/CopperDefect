#%%
import os
import os.path as osp
import pandas as pd
import json
os.chdir("/home/user/sun_chen/Projects/ZJDetection")

def read_json(json_path):
    with open(json_path,"r") as load_f:
        for line in load_f:
            log = json.loads(line.strip())    
            best_result = 0
            
            if "val" in log.values():
                val_result = log["mAP"]
                depth_result = log["RMSE"]
                recall_result = log["Recall"]
                precision_result = log["Precision"]
                f1_result = log["F1"]
                
                if val_result >= best_result:
                    best_result = val_result
                    best_depth = depth_result
                    best_recall = recall_result
                    best_precision = round(precision_result,4)
                    best_f1 = f1_result                    

    return best_result,best_depth,best_recall,best_precision,best_f1

#%%

result_dir = "./Work_dir/20220306/depth"
# result_dir = "./Work_dir/20220228/depth"



# for anchor in ["48","248"]:
metric_csv_list = []
for anchor in ["48"]:
    print(f"anchor_{anchor}:")
    for split in ["1x1","2x2","3x3"]:
        print(f"  split_{split}:")
        for method in os.listdir(osp.join(result_dir,f"anchor_{anchor}/{split}")):
            result_folder = osp.join(result_dir,f"anchor_{anchor}/{split}/{method}")
            result_list = [osp.join(result_folder,file_name) for file_name in os.listdir(result_folder) if "json" in file_name]
            for i in result_list:
                result = read_json(i)
                metric_info = [split,method]
                metric_info.extend(result)
                metric_csv_list.append(metric_info)
                print(f"    {method}:")
                print(f"        mAP: {result[0]}, RMSE: {result[1]}, Recall: {result[2]}, Precision: {result[3]}, F1: {result[4]}")    

metric_csv = pd.DataFrame(metric_csv_list)
metric_csv.to_csv("./results.csv")

# %%

# for anchor in ["48","248"]:
ratio_csv_list = []

for anchor in ["48"]:
    print(f"anchor:{anchor}")
    lambda_dir = f"./Work_dir/20220306/depth/lambda/anchor_{anchor}/3x3/frcn_hrnet_w40"
    for ratio in os.listdir(lambda_dir):
        result_folder = osp.join(lambda_dir,ratio)
        result_list = [osp.join(result_folder,file_name) for file_name in os.listdir(result_folder) if "json" in file_name]
        for i in result_list:
            result = read_json(i)
            ratio_info = [ratio]
            ratio_info.extend(result)
            ratio_csv_list.append(ratio_info)
            print(f"Ratio: {ratio}")
            print(f"mAP: {result[0]}, RMSE: {result[1]}, Recall: {result[2]}, Precision: {result[3]}, F1: {result[4]}")    

ratio_csv = pd.DataFrame(ratio_csv_list)
ratio_csv.to_csv("./results_lambda.csv")
# %%

