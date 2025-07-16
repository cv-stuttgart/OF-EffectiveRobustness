"""
Execute evaluate_effectiverobustness.py for multiple parameters.
"""
import argparse
import os

import subprocess
import json
import pandas as pd

from helper_functions.ownutilities import args_to_outputfilepath

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')

    parser.add_argument('--ckpt_csv')

    args = parser.parse_args()

    df_ckpts = pd.read_csv(args.ckpt_csv)

    failed_for = []

    for dset in [
        'FlyingThings3D',
        'Kitti15',
        'Sintel',
        'HD1KSplitScheurer',
        'Driving',
        'Viper',
        'Spring',
        'FlyingChairs',
    ]:
        for idx, row in df_ckpts.iterrows():
            net, wpath = row['net'], row['weight_path']
            
            
            # dataset_passes = ['clean', 'final'] if dset in ['Sintel', 'FlyingThings3D', 'Driving'] else ['""']
            dataset_passes = ['final'] if dset in ['Sintel', 'FlyingThings3D', 'Driving'] else ['']
            for dataset_pass in dataset_passes:
                    ds_type_arg = ('--dataset_pass', dataset_pass) if dataset_pass else tuple ()
                    if dset in ['FlyingChairs', 'FlyingThings3D', 'FlyingThings3DClean', 'FlyingThings3DFinal', 'SpringSplitScheurer', 'Viper']:
                        dataset_stage = ('--dataset_stage', 'validation')
                    else:
                        dataset_stage = ('--dataset_stage', 'training')
                    
                    
                    out_path = args_to_outputfilepath(
                        prefix='experiment_data',
                        net=net,
                        weight_path=wpath,
                        dataset=dset,
                        dataset_stage=dataset_stage[-1],
                        dataset_pass=dataset_pass,
                    )

                    if os.path.exists(out_path):
                        with open(out_path) as f:
                            x = json.load(f)
                            
                        # uncomment to skip already evaluated checkpoints 
                        # if x['config']['small_run'] == False and 'wauc_mean' in x:
                        #     pass
                    
                    if wpath != '':
                        custom_weight_path_argument = [
                            f'--custom_weight_path', wpath
                        ]
                    else:
                        custom_weight_path_argument = []

                    cmd = (
                        'python', 'evaluate_accuracy.py',
                        '--net', net,
                        '--dataset', dset,
                        *custom_weight_path_argument,
                        *dataset_stage,
                        *ds_type_arg,
                        # '--small_run'
                    )
                    print(' $$$ ', *cmd)
                    
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    for line in p.stdout:
                        print(line)
                    p.wait()
                    print('[system] return code', p.returncode)

                    if p.returncode != 0:
                        print(
                            f'[system] Failure for {(p.returncode, net, wpath, dset)}')
                        failed_for.append((p.returncode, net, wpath, dset, cmd))
                        # exit()

    if failed_for:
        for x in failed_for:
            print('The following model-dataset combinations failed: ', x[:-1])
            print('Try running the following command manually to check for errors:')
            print(" ".join(cmd))
