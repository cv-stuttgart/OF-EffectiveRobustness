
# First, update checkpoints_things.csv with all models and paths to be included in the baseline.
CKPT_CSV=config/checkpoints_things.csv


###################################
# Evaluate the accuracy of all models
#
# (This will take  vary long and may include debugging for each model architecture.
# You may want to directly call evaluate_accuracy.py for each model-dataset combination)
python -m scripts_evaluate.exec_evaluate_all --ckpt_csv $CKPT_CSV



###################################
# Fit the baseline to the pre-computed accuracies
#
# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Sintel --ood_dataset_stage training --ood_dataset_pass final \
    --output_folder paper_results

# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Kitti15 --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder paper_results

# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset HD1KSplitScheurer --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder paper_results

# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Driving --ood_dataset_stage training --ood_dataset_pass final \
    --output_folder paper_results

# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Viper --ood_dataset_stage validation --ood_dataset_pass "" \
    --output_folder paper_results

# fit the baseline to the WAUC of the models
python fit_baseline.py --ckpt_csv $CKPT_CSV \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Spring --ood_dataset_stage training --ood_dataset_pass "" \
    --output_folder paper_results
