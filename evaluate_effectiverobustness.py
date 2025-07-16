
import json
import os

from helper_functions import baselines

from helper_functions import parsing_file
from helper_functions.ownutilities import args_to_outputfilepath

if __name__ == '__main__':
    parser = parsing_file.create_parser(multi_model_csv=False, two_datasets=True, baseline_file=True)
    args = parser.parse_args()
    
    id_ds_name = args.id_dataset
    id_ds_stage = args.id_dataset_stage
    id_ds_pass = args.id_dataset_pass
    
    ood_ds_name = args.ood_dataset
    ood_ds_stage = args.ood_dataset_stage
    ood_ds_pass = args.ood_dataset_pass
    
    outfile = args_to_outputfilepath(
        prefix=args.output_folder,
        net=args.net,
        weight_path=args.custom_weight_path,
        dataset=id_ds_name,
        dataset_stage=id_ds_stage,
        dataset_pass=id_ds_pass,
    )
    with open(outfile, 'r') as f:
        id_metrics = json.load(f)
    id_wauc = id_metrics['wauc_mean']

    outfile = args_to_outputfilepath(
        prefix=args.output_folder,
        net=args.net,
        weight_path=args.custom_weight_path,
        dataset=ood_ds_name,
        dataset_stage=ood_ds_stage,
        dataset_pass=ood_ds_pass,
    )
    with open(outfile, 'r') as f:
        ood_metrics = json.load(f)
    ood_wauc = ood_metrics['wauc_mean']
    
    #
    # load baseline
    #
    baseline_file = args.baseline_file
    if not baseline_file:
        baseline_file = os.path.join(args.output_folder, f'baseline_{id_ds_name}_{ood_ds_name}.json')

    bl = baselines.BaselineLogitLinearRegression.deserialize_json(baseline_file)
    
    
    #
    # evaluate effective robustness
    #
    beta_wauc = bl.predict(id_wauc)[0]
    er = ood_wauc - beta_wauc
    
    print(f'Effective Robustness of {args.net} on {args.ood_dataset}: ER_WAUC = {er:.6f}')

    if er > 0:
        print("Congratz, you're above average!")
    else:
        print("This indicates bad generalization.")
