import os
import random
random.seed(3)

train_areas = ['area_1', 'area_2', 'area_4', 'area_6']
val_areas = ['area_3']
test_areas = ['area_5a', 'area_5b']

os.makedirs('./semi_supervised_configs', exist_ok=True)
os.makedirs('./semi_supervised_configs/train_config', exist_ok=True)

trainall = open('./configs/train_orig.txt').readlines()
testall = open('./configs/test_orig.txt').readlines()
print("Total original train images:", len(trainall))


#train-val split
def train_val_split(trainall, train_areas, val_areas):
    train_list = []
    val_list = []
    for line in trainall:
        area, name = line.strip().split(' ')
        if area in val_areas:
            val_list.append(line)
        elif area in train_areas:
            train_list.append(line)
        else:
            raise Exception("Area not in train or val")
    return train_list, val_list

train_list, val_list = train_val_split(trainall, train_areas, val_areas)
train_len = len(train_list)
val_len = len(val_list)


with open('./semi_supervised_configs/val.txt', 'w') as f:
    f.writelines(val_list)


ratios = [0.001, 0.002, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for ratio in ratios:
    denominator = int(1/ratio)
    labeled_len = int(len(train_list) * ratio)
    random.shuffle(train_list)
    labeled_list = train_list[:labeled_len]
    unlabeled_list = train_list[labeled_len:]
    
    with open('./final_data/configs/config_rgbd/train_labeled_1-{}.txt'.format(str(denominator)), 'w') as f:
        f.writelines(labeled_list)
    with open('./final_data/configs/config_rgbd/train_unlabeled_1-{}.txt'.format(str(denominator)), 'w') as f:
        f.writelines(unlabeled_list)