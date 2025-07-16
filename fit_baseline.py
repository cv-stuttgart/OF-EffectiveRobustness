
import argparse
import json
import os

import pandas as pd
from helper_functions import baselines

from helper_functions import parsing_file
from helper_functions.ownutilities import args_to_outputfilepath


if __name__ == '__main__':
    parser = parsing_file.create_parser(
        multi_model_csv=True,
        two_datasets=True,
        baseline_file=True
    )
    args = parser.parse_args()
    
    id_ds_name = args.id_dataset
    id_ds_stage = args.id_dataset_stage
    id_ds_pass = args.id_dataset_pass
    
    print(f'In-Distribution Dataset: {id_ds_name}, {id_ds_stage}, {id_ds_pass}')
    
    ood_ds_name = args.ood_dataset
    ood_ds_stage = args.ood_dataset_stage
    ood_ds_pass = args.ood_dataset_pass
    
    print(f'Out-of-Distribution Dataset: {ood_ds_name}, {ood_ds_stage}, {ood_ds_pass}')
    
    baseline_file = args.baseline_file
    if not baseline_file:
        baseline_file = os.path.join(args.output_folder, f'baseline_{id_ds_name}_{ood_ds_name}.json')
    
    df_ckpts = pd.read_csv(args.ckpt_csv)
    
    
    id_wauc = []
    ood_wauc = []
    
    #
    # read evaluation data
    #
    for idx, row in df_ckpts.iterrows():
        net, wpath = row['net'], row['weight_path']
        outfile = args_to_outputfilepath(
            prefix=args.output_folder,
            net=net,
            weight_path=wpath,
            dataset=id_ds_name,
            dataset_stage=id_ds_stage,
            dataset_pass=id_ds_pass,
        )
        with open(outfile, 'r') as f:
            id_metrics = json.load(f)
        id_wauc.append(id_metrics['wauc_mean'])
        
        
        outfile = args_to_outputfilepath(
            prefix=args.output_folder,
            net=net,
            weight_path=wpath,
            dataset=ood_ds_name,
            dataset_stage=ood_ds_stage,
            dataset_pass=ood_ds_pass,
        )
        
        with open(outfile, 'r') as f:
            ood_metrics = json.load(f)
        ood_wauc.append(ood_metrics['wauc_mean'])
    

    #
    # fit baseline
    #
    baseline = baselines.BaselineLogitLinearRegression()
    baseline.fit(id_wauc, ood_wauc)
    
    baseline.serialize_json(baseline_file)
    
    baselines.BaselineLogitLinearRegression.deserialize_json(baseline_file)
    print(f'--> Finished.')
