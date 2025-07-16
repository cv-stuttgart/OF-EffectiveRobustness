
NET="RAFT"
WPATH="models/_pretrained_weights/raft-things.pth"
ID_DATASET="FlyingThings3D"
ID_DATASET_STAGE="validation"
ID_DATASET_PASS="final"
OOD_DATASET="Kitti15"
OOD_DATASET_STAGE="training"
OOD_DATASET_PASS=""
OUTPUT_FOLDER="experiment_data"
BASELINE_FILE="paper_results/baseline_FlyingThings3D_Driving.json"

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset $ID_DATASET --dataset_stage $ID_DATASET_STAGE --dataset_pass $ID_DATASET_PASS \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset $OOD_DATASET --dataset_stage $OOD_DATASET_STAGE --dataset_pass $OOD_DATASET_PASS \
    --output_folder $OUTPUT_FOLDER

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset $OOD_DATASET --ood_dataset_stage $OOD_DATASET_STAGE --ood_dataset_pass $OOD_DATASET_PASS \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file $BASELINE_FILE
