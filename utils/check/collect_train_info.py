#%%
import os
import json
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ieee')
plt.style.context(['science','ieee','no-latex'])

def read_json(json_path):
    with open(json_path,"r") as load_f:
        json_file = json.load(load_f)
    return json_file

### img/bbox info collect

# train_anno_path = "/home/user/sun_chen/Projects/ZJDetection/Dataset/data/COCO-Annotations_depth/trainval.json"
train_anno_path = "/home/user/sun_chen/Projects/ZJDetection/Dataset/data/COCO-Annotations_depth/instance_all.json"

json_file = read_json(train_anno_path)

img_num = len(json_file["images"])
anno_num = len(json_file["annotations"])
print(f"imgs: {img_num}\n bbox: {anno_num}")

#%%
fig = plt.figure()

area_list = [ann_info["area"] for ann_info in json_file["annotations"]]
# relative_area_list = [area/(2428*2516)  for area in area_list]
relative_area_list = [area  for area in area_list]

depth_list = [ann_info["depth"] for ann_info in json_file["annotations"]]
depth_list.sort()
area_list.sort()

print(f"depth: {depth_list[0]}:{depth_list[-1]}")
print(f"area: {area_list[0]}:{area_list[-1]}")

plt.subplot(121)
plt.hist(depth_list,bins=100,facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("Depth")

# 显示纵轴标签
plt.ylabel("Frequency")
# 显示图标题
plt.title("Depth Distribution")
plt.subplot(122)
plt.hist(relative_area_list, bins=100,facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("Area")
plt.ylabel("Frequency")

# 显示图标题
plt.title("Area Distribution")
plt.tight_layout()
plt.show()
# plt.savefig
# %%
