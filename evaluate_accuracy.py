import datetime
import subprocess
import numpy as np
import torch

import os

from tqdm import tqdm
import json

from helper_functions import parsing_file
from helper_functions.config_specs import Conf
from helper_functions import ownutilities

from helper_functions.evaluation_loop import evaluate_model_on_dataset

if torch.cuda.is_available() and not Conf.config('useCPU'):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


START_TIME = datetime.datetime.now()

@torch.no_grad()
def evaluate_accuracy(model, args, output_filename: str):
    """ Evaluating the model on args.dataset """
    model.eval()
    
    output_path, fname = os.path.split(output_filename)

    print(f'stage {args.dataset_stage}')
    data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage,  # 'training',
                                                          dataset=args.dataset,
                                                          shuffle=False,
                                                          small_run=args.small_run,
                                                          dataset_pass=args.dataset_pass)
    assert has_gt, 'missing ground truth'
    test_dataset = data_loader.dataset
    output_path_timestamped = os.path.join(
        output_path, 'timestamped', args.creation_timestamp)
    output_filename_timestamped = os.path.join(output_path_timestamped, fname)

    print(f"Write results to '{output_path_timestamped}'")


    metrics, global_metrics = evaluate_model_on_dataset(
        model,
        test_dataset,
        net=args.net,
        device=device,
    )

    serializable_result = {
        'config': dict(args._get_kwargs())
    }
    # Don't store metric for each frame pair -> too many numbers for our poor little storage
    # serializable_result.update(
    #     {   # metrics for each framepair
    #         f'{key}_list': [
    #             entry[key] if isinstance(entry[key], (int, float))
    #             else float(entry[key])
    #             for entry in metrics
    #         ]
    #         for key in metrics[0]
    #     }
    # )
    serializable_result.update(
        {  # mean of metrics calculated independently per frame pair
            f'{key}_mean': float(np.mean([entry[key] for entry in metrics]))
            for key in metrics[0]
        }
    )
    serializable_result.update(
        {  # metrics over all pixels
            f'{key}_all': val if isinstance(val, (int, float)) else float(val)
            for key, val in global_metrics.items()
        }
    )
    runtime_delta = (datetime.datetime.now() - START_TIME)
    serializable_result['runtime_s'] = runtime_delta.total_seconds()
    serializable_result['dataset_size'] = len(data_loader)

    # save result
    if not os.path.exists(output_path_timestamped):
        os.makedirs(output_path_timestamped)

    # write with timestamp
    with open(output_filename_timestamped, 'w') as f:
        json.dump(serializable_result, f, indent=3)

    if not args.small_run:
        # (over)write file without timestamp
        if os.path.exists(output_filename):
            print(f'Overwriting {output_filename}')
        else:
            print(f'Write to {output_filename}')

        serializable_result = {  # don't store per-frame results
            k: v for (k, v) in serializable_result.items() if not k.endswith('_list')
        }
        with open(output_filename, 'w') as f:
            json.dump(serializable_result, f, indent=3)
    print()
    # print(aees)
    print(f'(AEPE, WAUC, fl-1px, fl-3px, fl-5px) on {args.dataset} using {args.weight_path}',
          serializable_result['aee_mean'],
          serializable_result['wauc_mean'],
          serializable_result['fl-epe-1px_mean'],
          serializable_result['fl-epe-3px_mean'],
          serializable_result['fl-epe-5px_mean']
          )


if __name__ == '__main__':
    parser = parsing_file.create_parser()
    args = parser.parse_args()
    print(f'--> Beginning evaluation for {args.net} on {args.dataset}.')

    # load model
    model, path_weights = ownutilities.import_and_load(
        net=args.net, device=device, custom_weight_path=args.custom_weight_path
    )

    print('path_weights', path_weights)
    if args.custom_weight_path:
        assert path_weights == args.custom_weight_path

    args.weight_path = path_weights
    args.creation_timestamp = START_TIME.strftime(r'%y%m%d-%H%M%S')

    output_filename = ownutilities.args_to_outputfilepath(
        args.output_folder,
        net=args.net,
        weight_path=path_weights,
        dataset=args.dataset,
        dataset_pass=args.dataset_pass,
        dataset_stage=args.dataset_stage
    )
    
    evaluate_accuracy(model, args,
                      output_filename=output_filename)

    print(f'--> Finished.')
