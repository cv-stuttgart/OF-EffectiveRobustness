
NET="RAFT"
WPATH="models/_pretrained_weights/raft-things.pth"
# OUTPUT_FOLDER is the folder where the accuracy of the model is stored.
OUTPUT_FOLDER="experiment_data"

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset FlyingThings3D --dataset_stage validation --dataset_pass "final" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset Sintel --dataset_stage training --dataset_pass "final" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset Kitti15 --dataset_stage training --dataset_pass "" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset HD1KSplitScheurer --dataset_stage training --dataset_pass "" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset Driving --dataset_stage training --dataset_pass "" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset Viper --dataset_stage validation --dataset_pass "" \
    --output_folder $OUTPUT_FOLDER

python evaluate_accuracy.py \
    --net $NET --custom_weight_path $WPATH \
    --dataset Spring --dataset_stage training --dataset_pass "" \
    --output_folder $OUTPUT_FOLDER
