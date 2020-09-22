#!/bin/bash

## 1) [Required] Please specify the path to the PSCAL VOC 2012 dataset (e.g., ~/datasets/VOCdevkit/VOC2012/)
dataset_folder=path_to_dataset_folder
## 2) [Required] Please specify the path to the folder to save the feature/label/weight (e.g., ./save)
save_folder=path_to_the_folder_for_saving_output_results
## 3) Please specify the path to the pretrained weight (It is default to save in the folder named weights)
pretrained_model=./weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params

## 4) Please specify the path to the model for each round
R1_extractor_model=./weights/res38_cls.pth
R2_extractor_model=${save_folder}/weight/k10_R1_resnet_cls_ep30.pth
R3_extractor_model=${save_folder}/weight/k10_R2_resnet_cls_ep40.pth
final_model=${save_folder}/weight/k10_R3_resnet_cls_ep50.pth \

## 5) Please specify the path to save the generated response map
save_cam_folder=${save_folder}/final_result




## R1
# Extract the feature
python extract_feature.py --weights ${R1_extractor_model} --infer_list ./voc12/train_aug.txt --voc12_root ${dataset_folder} --save_folder ${save_folder} --k_cluster 10 --from_round_nb 0
# Generate the pseudo label
python create_pseudo_label.py --save_folder ${save_folder} --k_cluster 10 --for_round_nb 1
# Train the classifier
python train_cls.py --lr 0.01 --voc12_root ${dataset_folder} --weights ${pretrained_model} --round_nb 1 --save_folder ${save_folder}



## R2
# Extract the feature
python extract_feature.py --weights ${R2_extractor_model} --infer_list ./voc12/train_aug.txt --voc12_root ${dataset_folder} --save_folder ${save_folder} --k_cluster 10 --from_round_nb 1
# Generate the pseudo label
python create_pseudo_label.py --save_folder ${save_folder} --k_cluster 10 --for_round_nb 2
# Train the classifier
python train_cls.py --lr 0.01 --voc12_root ${dataset_folder} --weights ${pretrained_model} --round_nb 2 --save_folder ${save_folder}



## R3
# Extract the feature
python extract_feature.py --weights ${R3_extractor_model} --infer_list ./voc12/train_aug.txt --voc12_root ${dataset_folder} --save_folder ${save_folder} --k_cluster 10 --from_round_nb 2
# Generate the pseudo label
python create_pseudo_label.py --save_folder ${save_folder} --k_cluster 10 --for_round_nb 3
# Train the classifier
python train_cls.py --lr 0.01 --voc12_root ${dataset_folder} --weights ${pretrained_model} --round_nb 3 --save_folder ${save_folder}



## Infer the classifier and generate the response map with the final model
python infer_cls.py --infer_list voc12/train.txt --voc12_root ${dataset_folder} --weights ${final_model} --save_path ${save_cam_folder} --save_out_cam 1 --k_cluster 10 --round_nb 3
