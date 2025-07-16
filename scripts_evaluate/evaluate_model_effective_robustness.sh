
NET="RAFT"
WPATH="models/_pretrained_weights/raft-things.pth"
ID_DATASET="FlyingThings3D"
ID_DATASET_STAGE="validation"
ID_DATASET_PASS="final"
# OUTPUT_FOLDER is the folder where the accuracy of the model is stored.
# evaluate_effectiverobustness.py does not write any output to disk.
OUTPUT_FOLDER="experiment_data"

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset Sintel --ood_dataset_stage training --ood_dataset_pass "final" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_Sintel.json

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset Kitti15 --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_Kitti15.json

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset HD1KSplitScheurer --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_HD1KSplitScheurer.json

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset Driving --ood_dataset_stage training --ood_dataset_pass "final" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_Driving.json

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset Viper --ood_dataset_stage validation --ood_dataset_pass "" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_Viper.json

python evaluate_effectiverobustness.py \
    --net $NET --custom_weight_path $WPATH \
    --id_dataset $ID_DATASET --id_dataset_stage $ID_DATASET_STAGE --id_dataset_pass $ID_DATASET_PASS \
    --ood_dataset Spring --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder $OUTPUT_FOLDER \
    --baseline_file paper_results/baseline_FlyingThings3D_Spring.json
