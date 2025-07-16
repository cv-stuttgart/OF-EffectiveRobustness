'''Execute each model on the first image pair from KITTI.'''

from argparse import Namespace
import datetime
from glob import glob
import subprocess
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch

import os
import os.path as osp

from tqdm import tqdm
import json

from helper_functions import datasets
from helper_functions import parsing_file
from helper_functions.config_specs import Conf, Paths
from helper_functions import ownutilities

from flow_library.flow_show import getFlowVis

from helper_functions.evaluation_loop import (
    _combine_metrics,
    eval_metrics_flat,
    evaluate_model_on_dataset,
)

if torch.cuda.is_available() and not Conf.config("useCPU"):
    print("GPU available")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


START_TIME = datetime.datetime.now()


class RobustKITTI(datasets.FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="/data2/katrin/datasets/RobustKITTI/Kitti15_pixelate/*/pixelate",
        has_gt=False,
    ):
        super(RobustKITTI, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt

        images1 = sorted(glob(osp.join(root, "image_2/*_10.png")))
        images2 = sorted(glob(osp.join(root, "image_2/*_11.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))

        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the KITTI dataset."
                % root
            )

        assert (
            len(self.image_list) == 200
        ), f"Expected 200 images, but found {len(self.image_list)}. Check the dataset path: {root}"


class WeatherKITTI(datasets.FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="/data2/katrin/datasets/KITTI_weather_rain_15",
        has_gt=False,
    ):
        super(WeatherKITTI, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt

        images1 = sorted(glob(osp.join(root, "*/*.png1*.png")))
        images2 = sorted(glob(osp.join(root, "*/*.png2*.png")))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split("/")[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, "flow_occ/*_10.png")))

        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the KITTI dataset."
                % root
            )

        assert (
            len(self.image_list) == 200
        ), f"Expected 200 images, but found {len(self.image_list)}. Check the dataset path: {root}"


class RobustSintel(datasets.FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split=Paths.splits("sintel_train"),
        root=Paths.config("sintel_mpi"),
        dataset_pass="clean",
        has_gt=False,
        scenes=None,
        every_nth_img=1,
    ):
        super(RobustSintel, self).__init__(aug_params)
        if dataset_pass not in ["clean", "final"]:
            raise ValueError(
                "dataset_pass must be either 'clean' or 'final', but got %s"
                % dataset_pass
            )

        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dataset_pass)

        self.has_gt = False  # has_gt

        for scene in sorted(os.listdir(image_root)):
            if scenes is None or scene in scenes:
                image_list = sorted(glob(osp.join(image_root, scene, "*.png")))
                for i in range(len(image_list) - 1):
                    self.image_list += [[image_list[i], image_list[i + 1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if self.has_gt:
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

        if every_nth_img > 1:
            self.image_list = self.image_list[::every_nth_img]
            self.extra_info = self.extra_info[::every_nth_img]
            self.flow_list = self.flow_list[::every_nth_img]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the MPI Sintel dataset."
                % root
            )


class WeatherSintel(datasets.FlowDataset):
    def __init__(
        self,
        aug_params=None,
        split=Paths.splits("sintel_train"),
        root=Paths.config("sintel_mpi"),
        dataset_pass="clean",
        has_gt=False,
        scenes=None,
        every_nth_img=1,
    ):
        super(WeatherSintel, self).__init__(aug_params)
        if dataset_pass not in ["clean", "final"]:
            raise ValueError(
                "dataset_pass must be either 'clean' or 'final', but got %s"
                % dataset_pass
            )

        flow_root = osp.join(root, split, "flow")
        image_root = osp.join(root, split, dataset_pass)

        self.has_gt = False  # has_gt

        for scene in sorted(os.listdir(image_root)):
            if scenes is None or scene in scenes:
                first_frames = sorted(glob(osp.join(image_root, scene, "*_1_*.png")))
                second_frames = sorted(glob(osp.join(image_root, scene, "*_2_*.png")))
                # image_list = sorted(glob(osp.join(image_root, scene, "*.png")))

                for i, (img1, img2) in enumerate(zip(first_frames, second_frames)):
                    frame_id = img1.split("/")[-1]
                    self.image_list += [[img1, img2]]
                    self.extra_info += [(scene, i)]

                # for i in range(len(image_list) - 1):
                #     self.image_list += [[image_list[i], image_list[i + 1]]]
                #     self.extra_info += [(scene, i)]  # scene and frame_id

                if self.has_gt:
                    self.flow_list += sorted(glob(osp.join(flow_root, scene, "*.flo")))

        if every_nth_img > 1:
            self.image_list = self.image_list[::every_nth_img]
            self.extra_info = self.extra_info[::every_nth_img]
            self.flow_list = self.flow_list[::every_nth_img]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the MPI Sintel dataset."
                % root
            )


def store_results(
    args,
    metrics,
    global_metrics,
    output_filename: str = None,
    dataset_size=None,
    output_filename_timestamped: str = None,
):
    assert output_filename is not None, "output_filename must be set"

    serializable_result = {"config": dict(args._get_kwargs())}
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
            f"{key}_mean": float(np.mean([entry[key] for entry in metrics]))
            for key in metrics[0]
        }
    )
    serializable_result.update(
        {  # metrics over all pixels
            f"{key}_all": val if isinstance(val, (int, float)) else float(val)
            for key, val in global_metrics.items()
        }
    )
    runtime_delta = datetime.datetime.now() - START_TIME
    serializable_result["runtime_s"] = runtime_delta.total_seconds()
    serializable_result["dataset_size"] = dataset_size

    print()
    print(
        f"(AEPE, WAUC, fl-1px, fl-3px, fl-5px) on {args.dataset} using {args.weight_path}",
        serializable_result["aee_mean"],
        serializable_result["wauc_mean"],
        serializable_result["fl-epe-1px_mean"],
        serializable_result["fl-epe-3px_mean"],
        serializable_result["fl-epe-5px_mean"],
    )

    if output_filename_timestamped is not None:
        output_path_timestamped, fname = os.path.split(output_filename_timestamped)
        # save result
        if not os.path.exists(output_path_timestamped):
            os.makedirs(output_path_timestamped)

        # write with timestamp
        with open(output_filename_timestamped, "w") as f:
            json.dump(serializable_result, f, indent=3)

    else:
        # write without timestamp
        output_path, fname = os.path.split(output_filename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    if not args.small_run:
        # (over)write file without timestamp
        if os.path.exists(output_filename):
            print(f"Overwriting {output_filename}")
        else:
            print(f"Write to {output_filename}")

        serializable_result = {  # don't store per-frame results
            k: v for (k, v) in serializable_result.items() if not k.endswith("_list")
        }
        with open(output_filename, "w") as f:
            json.dump(serializable_result, f, indent=3)


@torch.no_grad()
def evaluate_model_on_one_data_instance(model, args, output_filename: str):
    """Evaluating the model on args.dataset"""
    model.eval()
    net = args.net

    ############
    # Datasets #
    ############
    data_loader, has_gt = ownutilities.prepare_dataloader(
        args.dataset_stage,  # 'training',
        dataset=args.dataset,
        shuffle=False,
        small_run=args.small_run,
        dataset_pass=args.dataset_pass
    )
    clean_dataset = data_loader.dataset

    ###################
    # Evaluation Loop #
    ###################
    intermediate_results = []

    for i in tqdm(range(min(len(clean_dataset), 2))):
        img1_clean, img2_clean, _flow_gt, _valid = clean_dataset[i]

        padder, [img1_clean, img2_clean] = (
            ownutilities.preprocess_img(
                net, img1_clean, img2_clean
            )
        )
        img1_clean = img1_clean[None].to(device)
        img2_clean = img2_clean[None].to(device)

        flow_pr_clean = ownutilities.compute_flow(
            model, net, img1_clean, img2_clean
        )

        [flow_pr_clean] = ownutilities.postprocess_flow(
            net, padder, flow_pr_clean
        )

        
        ownutilities.quickvis_flow(
            flow_pr_clean, filename=output_filename.replace("metrics.json", f"{i:04d}_flow.png")
        )
        ownutilities.quickvis_tensor(
            img1_clean[0],
            filename=output_filename.replace("metrics.json", f"{i:04d}_img1.png"),
        )
        ownutilities.quickvis_tensor(
            img2_clean[0],
            filename=output_filename.replace("metrics.json", f"{i:04d}_img2.png"),
        )
    # end evaluation loop


def main(args):
    print(f"--> Beginning evaluation for {args.net} on {args.dataset}.")

    # load model
    model, path_weights = ownutilities.import_and_load(
        net=args.net, device=device, custom_weight_path=args.custom_weight_path
    )
    if args.custom_weight_path:
        assert path_weights == args.custom_weight_path

    args.weight_path = path_weights
    args.creation_timestamp = START_TIME.strftime(r"%y%m%d-%H%M%S")

    output_filename = ownutilities.args_to_outputfilepath(
        args.output_folder,
        net=args.net,
        weight_path=path_weights,
        dataset=args.dataset,
        dataset_pass=args.dataset_pass,
        dataset_stage=args.dataset_stage,
    )

    evaluate_model_on_one_data_instance(model, args=args, output_filename=output_filename)

    print(f"--> Finished evaluation for {args.net} on {args.dataset}.")

if __name__ == "__main__":
    parser = parsing_file.create_parser()
    args = parser.parse_args()
    
    ckpts = pd.read_csv(
        "/home/bauerkn/programming/FlowUnderAttackMerged/docs/checkpoints_things.csv"
    )
    weight_pairs = list(zip(ckpts["net"], ckpts["weight_path"]))
    
    working_models = []
    failing_models = []
    
    for net, weight_path in weight_pairs:
        try:
            args.net = net
            args.custom_weight_path = weight_path

            main(args)
            
            working_models.append(net)
        
        except Exception as e:
            print(
                f"Error during evaluation for net {net}: {type(e)} {e}"
            )
            failing_models.append(net)
            continue
    
    print(f"Working models: {working_models}")
    print(f"Failed models: {failing_models}")
