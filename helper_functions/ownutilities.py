
from helper_functions.config_specs import Paths, Conf
from helper_functions import datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image
import json
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from argparse import Namespace
import os
import sys
# required to prevent ModuleNotFoundError for 'flow_plot'. The flow_library is a submodule, which imports its own functions and can therefore not be imported with flow_library.flow_plot
if True:
    sys.path.append("flow_library")
    from flow_plot import colorplot_light


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


class InputPadder:
    """Pads images such that dimensions are divisible by divisor

    This method is taken from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    """

    def __init__(self, dims, divisor=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd //
                         2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        """Pad a batch of input images such that the image size is divisible by the factor specified as divisor

        Returns:
            list: padded input images
        """
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def get_dimensions(self):
        """get the original spatial dimension of the image

        Returns:
            int: original image height and width
        """
        return self.ht, self.wd

    def unpad(self, x):
        """undo the padding and restore original spatial dimension

        Args:
            x (tensor): a tensor with padded dimensions

        Returns:
            tesnor: tensor with removed padding (i.e. original spatial dimension)
        """
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
    """import a model and load pretrained weights for it

    Args:
        net (str, optional):
            the desired network to load. Defaults to 'RAFT'.
        make_unit_input (bool, optional):
            model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
        variable_change (bool, optional):
            apply change of variables (COV). Defaults to False.
        device (torch.device, optional):
            changes the selected device. Defaults to torch.device("cpu").
        make_scaled_input_model (bool, optional):
            load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

    Raises:
        RuntimeWarning: Unknown model type

    Returns:
        torch.nn.Module: PyTorch optical flow model with loaded weights
    """

    if make_unit_input == True or variable_change == True or make_scaled_input_model:
        from helper_functions.own_models import ScaledInputModel
        model = ScaledInputModel(net, make_unit_input=make_unit_input,
                                 variable_change=variable_change, device=device, **kwargs)
        print("--> transforming model to 'make_unit_input'=%s, 'variable_change'=%s\n" %
              (str(make_unit_input), str(variable_change)))
        path_weights = model.return_path_weights()

    else:
        model = None
        path_weights = ""
        custom_weight_path = kwargs["custom_weight_path"] if "custom_weight_path" in kwargs else ""
        try:
            if net == 'RAFT':
                from models.raft.raft import RAFT

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/raft-sintel.pth'

                # possible adjustements to the config can be made in the file
                # found under models/_config/raft_config.json
                with open("models/_config/raft_config.json") as file:
                    config = json.load(file)

                model = torch.nn.DataParallel(RAFT(config))
                # load pretrained weights
                model.load_state_dict(torch.load(
                    path_weights, map_location=device))

            elif net == 'GMA':
                from models.gma.network import RAFTGMA

                # set the path to the corresponding weights for initializing the model
                path_weights = custom_weight_path or 'models/_pretrained_weights/gma-sintel.pth'

                # possible adjustements to the config file can be made
                # under models/_config/gma_config.json
                with open("models/_config/gma_config.json") as file:
                    config = json.load(file)
                    # GMA accepts only a Namespace object when initializing
                    config = Namespace(**config)

                model = torch.nn.DataParallel(RAFTGMA(config))

                model.load_state_dict(torch.load(
                    path_weights, map_location=device))

            elif net == "FlowFormer":
                from models.FlowFormer.core.FlowFormer import build_flowformer
                from models.FlowFormer.configs.things_eval import get_cfg as get_things_cfg

                path_weights = custom_weight_path or 'models/_pretrained_weights/flowformer_weights/sintel.pth'
                cfg = get_things_cfg()
                model_args = Namespace(
                    model=path_weights, mixed_precision=False, alternate_corr=False)
                cfg.update(vars(model_args))

                model = torch.nn.DataParallel(build_flowformer(cfg))
                model.load_state_dict(torch.load(
                    cfg.model, map_location=torch.device('cpu')))

            elif net == "VideoFlowBOF":
                from models.VideoFlow.core.Networks import build_network
                from models.VideoFlow.configs.things import get_cfg as get_things_cfg

                path_weights = custom_weight_path or 'models/_pretrained_weights/VideoFlow_ckpt/BOF_sintel.pth'
                cfg = get_things_cfg()
                model_args = Namespace(
                    model=path_weights, mixed_precision=False, alternate_corr=False)
                cfg.update(vars(model_args))

                model = torch.nn.DataParallel(build_network(cfg))
                model.load_state_dict(torch.load(
                    cfg.model, map_location=torch.device('cpu')))
                print('VideoFlowBOF')

            elif net == "MemFlow":
                from models.MemFlow.core.Networks import build_network
                from models.MemFlow.configs.things_memflownet import get_cfg

                path_weights = custom_weight_path or 'models/_pretrained_weights/memflow_weights/MemFlowNet_sintel.pth'

                cfg = get_cfg()
                model_args = Namespace(
                    restore_ckpt=path_weights, mixed_precision=False)
                cfg.update(vars(model_args))

                # rank = cfg.node_rank * cfg.gpus + gpu
                # torch.cuda.set_device(rank)
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    build_network(cfg)).cuda()
                # , device_ids=[rank] # not sure what rank is...
                model = torch.nn.parallel.DataParallel(model)

                # load weights
                print("[Loading ckpt from {}]".format(cfg.restore_ckpt))
                ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
                ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt
                if 'module' in list(ckpt_model.keys())[0]:
                    model.load_state_dict(ckpt_model, strict=True)
                else:
                    model.module.load_state_dict(ckpt_model, strict=True)

                # [ugly hack] store config in model
                model.config = cfg

            elif net == 'SKFlow':
                from models.SKFlow.core.models import SK_Decoder

                path_weights = custom_weight_path or 'models/_pretrained_weights/skflow_weights/skflow-sintel.pth'

                # command line arguments for skflow / copied from SKFlow/scripts/infer.sh or - if undefined there - default arguments from evaluate.py
                sk_args = Namespace(
                    num_heads=1,
                    mixed_precision=False,
                    position_only=False,
                    position_and_content=False,
                    UpdateBlock='SKUpdateBlock6_Deep_nopoolres_AllDecoder',
                    k_conv=(1, 15),
                    PCUpdater_conv=(1, 7)
                )
                model = torch.nn.DataParallel(SK_Decoder(sk_args))
                model.load_state_dict(torch.load(path_weights))

            elif net == 'PWCNet':
                from models.PWCNet.PWCNet import PWCDCNet

                # set path to pretrained weights:
                path_weights = custom_weight_path or 'models/_pretrained_weights/pwc_net_chairs.pth.tar'
                with warnings.catch_warnings():
                    # this will catch the deprecated warning for spynet and pwcnet to avoid messy console
                    warnings.simplefilter("ignore", UserWarning)
                    model = PWCDCNet()

                weights = torch.load(path_weights, map_location=device)
                if 'state_dict' in weights.keys():
                    model.load_state_dict(weights['state_dict'])
                else:
                    model.load_state_dict(weights)
                model.to(device)

            elif net == 'SpyNet':
                from models.SpyNet.SpyNet import Network as SpyNet
                # weights for SpyNet are loaded during initialization
                model = SpyNet(nlevels=6, pretrained=True)
                model.to(device)

            elif net[:8] == "FlowNet2":
                # hard coding configuration for FlowNet2
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNet2":
                    from models.FlowNet.FlowNet2 import FlowNet2
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
                    model = FlowNet2(args_fn, div_flow=20, batchNorm=False)

                elif net == "FlowNet2S":
                    from models.FlowNet.FlowNet2S import FlowNet2S
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2-S_checkpoint.pth.tar'
                    model = FlowNet2S(args_fn, div_flow=20, batchNorm=False)

                elif net == "FlowNet2C":
                    from models.FlowNet.FlowNet2C import FlowNet2C
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/FlowNet2-C_checkpoint.pth.tar'
                    model = FlowNet2C(args_fn, div_flow=20, batchNorm=False)

                else:
                    raise ValueError("Unknown FlowNet2 type: %s" % (net))

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            elif net == "FlowNetCRobust":
                from models.FlowNetCRobust.FlowNetC_flexible_larger_field import FlowNetC_flexible_larger_field

                # initialize model and load pretrained weights
                path_weights = custom_weight_path or 'models/_pretrained_weights/RobustFlowNetC.pth'
                model = FlowNetC_flexible_larger_field(
                    kernel_size=3, number_of_reps=3)

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights)

                model.to(device)

            elif net[:7] == "FlowNet":
                # hard coding configuration for FlowNet
                args_fn = Namespace(fp16=False, rgb_max=255.0)

                if net == "FlowNetC":
                    from models.FlowNet.FlowNetC import FlowNetC
                    # set path to pretrained weights
                    path_weights = custom_weight_path or 'models/_pretrained_weights/flownetc_EPE1.766.tar'
                    model = FlowNetC(args_fn, div_flow=20, batchNorm=False)

                else:
                    raise ValueError("Unknown FlowNet type: %s" % (net))

                weights = torch.load(path_weights, map_location=device)
                model.load_state_dict(weights['state_dict'])

                model.to(device)

            elif net.startswith("unimatch"):
                from models.unimatch.unimatch.unimatch import UniMatch


                # different checkpoints need different hyper arguments ...
                kwargs_unimatch = dict(
                    feature_channels=128,
                    num_scales=1, # Hierarchical Matching Refinement
                    upsample_factor=8,
                    num_head=1,
                    ffn_dim_expansion=4,
                    num_transformer_layers=6,
                    reg_refine=False, # Local Regression Refinement
                    task='flow'
                )

                if net == 'unimatch':
                    path_weights = custom_weight_path or 'models/_pretrained_weights/unimatch_weights/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
                    if 'regrefine' in path_weights:
                        kwargs_unimatch.update(dict(
                            reg_refine=True,
                        ))
                    if 'scale2' in path_weights:
                        kwargs_unimatch.update(dict(
                            upsample_factor=4,
                            num_scales=2,
                        ))
                else:
                    if 'regrefine' in net:
                        kwargs_unimatch.update(dict(
                            reg_refine=True,
                        ))
                        
                    if 'scale2' in net:
                        kwargs_unimatch.update(dict(
                            upsample_factor=4,
                            num_scales=2,
                        ))
                    
                    if 'regrefine' in net:
                        path_weights = custom_weight_path or 'models/_pretrained_weights/unimatch_weights/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth'
                    elif 'scale2' in net:
                        path_weights = custom_weight_path or 'models/_pretrained_weights/unimatch_weights/gmflow-scale2-mixdata-train320x576-9ff1c094.pth'
                    elif 'scale1' in net:
                        path_weights = custom_weight_path or 'models/_pretrained_weights/unimatch_weights/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth'
                    else:
                        print('Invalid unimatch net choice', net)

                checkpoint = torch.load(path_weights, map_location=device)

                # default values of the argument parser
                model = UniMatch(**kwargs_unimatch).to(device)
                model = torch.nn.DataParallel(model)
                model_without_ddp = model.module

                model_without_ddp.load_state_dict(checkpoint['model'])

            elif net[:8] == 'SEA-RAFT':
                from models.SEA_RAFT.core.raft import RAFT as SEA_RAFT
                from models.SEA_RAFT.config.parser import json_to_args

                if net.endswith('-L'):
                    # SEA-RAFT(L) is equal to (M) except that it uses iters=12
                    path_weights = custom_weight_path or 'models/_pretrained_weights/SEA-RAFT-models/Tartan-C-T-TSKH432x960-M.pth'
                    args = json_to_args(
                        'models/SEA_RAFT/config/eval/sintel-L.json')
                if net.endswith('-M'):
                    # SEA-RAFT(M) uses first 13 layers of ResNet-34, iters=4
                    path_weights = custom_weight_path or 'models/_pretrained_weights/SEA-RAFT-models/Tartan-C-T-TSKH432x960-M.pth'
                    args = json_to_args(
                        'models/SEA_RAFT/config/eval/sintel-M.json')
                if net.endswith('-S'):
                    # SEA-RAFT(S) uses first 6 layers of ResNet-18
                    path_weights = custom_weight_path or 'models/_pretrained_weights/SEA-RAFT-models/Tartan-C-T-TSKH432x960-S.pth'
                    args = json_to_args(
                        'models/SEA_RAFT/config/eval/sintel-S.json')

                model = SEA_RAFT(args)
                state_dict = torch.load(path_weights)
                model.load_state_dict(state_dict, strict=False)
                model.to(device)

            elif net == 'MS-RAFT+':
                from models.ms_raft_plus.MS_RAFT_plus import MS_RAFT_plus
                config = dict(
                    mixed_precision=False,
                    lookup=dict(
                        pyramid_levels=2,
                        radius=4,
                    ),
                    cuda_corr=True,
                )
                model = torch.nn.DataParallel(MS_RAFT_plus(config))
                path_weights = custom_weight_path or 'models/_pretrained_weights/ms_raft_plus/mixed.pth'

                model.load_state_dict(torch.load(path_weights))
                model.module.to(device)
                model.to(device)

            elif net == "CCMR+":
                from models.CCMR.core.CCMR import CCMR

                config = dict(
                    model_type = "CCMR+",
                    num_scales=4,
                    mixed_precision=True,
                    fnet_norm = "group",
                    cnet_norm = "group",
                )
                
                model = torch.nn.DataParallel(CCMR(config))
                path_weights = custom_weight_path or 'models/_pretrained_weights/ccmr_weights/CCMR+_sintel.pth'
                model.load_state_dict(torch.load(path_weights))
                model.to(device)

            elif net == "MatchFlowG":
                from models.MatchFlow.core.network import RAFTGMA, RAFTGMA_QuadtreeFnet
                from models.MatchFlow.core.common import torch_init_model

                from models.MatchFlow.configs.default import get_cfg

                mf_cfg = get_cfg()
                mf_cfg.raft = False
                mf_cfg.image_size = [416, 736]
                mf_cfg.num_heads = 1
                mf_cfg.position_and_content = False
                mf_cfg.position_only = False
                mf_cfg.mixed_precision = False
                mf_cfg.matching_model_path = './models/_pretrained_weights/matchflow_weights/outdoor.ckpt'

                model = torch.nn.DataParallel(RAFTGMA_QuadtreeFnet(mf_cfg))
                path_weights = custom_weight_path or './models/_pretrained_weights/matchflow_weights/matchflow-g/matchflow-g-sintel.pth'
                torch_init_model(model, torch.load(
                    path_weights, map_location='cpu'), key='model')

                model.to(device)

            elif net == "MatchFlowR":
                from models.MatchFlow.core.network import RAFTGMA, RAFTGMA_QuadtreeFnet
                from models.MatchFlow.core.common import torch_init_model

                from models.MatchFlow.configs.default import get_cfg

                mf_cfg = get_cfg()
                mf_cfg.raft = True
                mf_cfg.image_size = [416, 736]
                mf_cfg.num_heads = 1
                mf_cfg.position_and_content = False
                mf_cfg.position_only = False
                mf_cfg.mixed_precision = False
                mf_cfg.matching_model_path = './models/_pretrained_weights/matchflow_weights/outdoor.ckpt'

                model = torch.nn.DataParallel(RAFTGMA_QuadtreeFnet(mf_cfg))
                path_weights = custom_weight_path or './models/_pretrained_weights/matchflow_weights/matchflow-r/matchflow-r-things.pth'
                torch_init_model(model, torch.load(
                    path_weights, map_location='cpu'), key='model')

                model.to(device)
            
            elif net == 'IRR-PWC':
                from models.irr.tools import instance_from_kwargs
                from models.irr.models.IRR_PWC import PWCNet
                # from models.irr.configuration import configure_checkpoint_saver

                # I don't know which hyperparameters IRR has but it seems to run without.
                kwargs = dict()
                kwargs['args'] = Namespace(**kwargs)
                model = instance_from_kwargs(PWCNet, kwargs)

                path_weights = custom_weight_path or './models/_pretrained_weights/irr_weights/pwcnet/IRR-PWC_things3d/checkpoint_best.ckpt'
        
                checkpoint_with_state = torch.load(path_weights)
                print(list(checkpoint_with_state.keys()))
                # strip the prefix '_model.' from keys
                state_dict = {
                    key[len('_model.'):]: value for key, value in checkpoint_with_state['state_dict'].items()
                }
                model.load_state_dict(state_dict)
                # args = Namespace(
                #     checkpoint=path_weights,
                #     checkpoint_include_params=['*'],
                #     checkpoint_exclude_params=[],
                # )
                # checkpoint_saver, checkpoint_stats = configure_checkpoint_saver(args, model)

                model.to(device)

            elif net[:8] == "ptlflow-":
                # raise NotImplementedError()
                sys.path.append("models/ptlflow")
                from ptlflow import get_model_reference, get_model
                
                # 'sintel' loads sintel model from torch hub
                path_weights = custom_weight_path or 'sintel'
                model_name = net[8:]
                
                if path_weights in ['chairs', 'things', 'sintel', 'kitti']:
                    # load weights from hub
                    model = get_model(model_name, ckpt_path=path_weights)
                else:
                    state_dict = torch.load(path_weights)
                    state_dict_cpy = {}
                    for key in state_dict:
                        state_dict_cpy['.'.join(key.split('.')[1:])] = state_dict[key]
                    # print(list(state_dict.keys()))
                    model.load_state_dict(state_dict_cpy)
                model.to(device)

            if model is None:
                raise RuntimeWarning(
                    'The network %s is not a valid model option for import_and_load(network). No model was loaded. Use "RAFT", "GMA", "FlowNetC", "PWCNet" or "SpyNet" instead.' % (net))
        except FileNotFoundError as e:
            print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
            exit()

        print("--> flow network is set to %s" % net)
    return model, path_weights


def prepare_dataloader(mode='training', dataset='Sintel', batch_size=1, shuffle=False, small_run=False, dataset_pass=''):
    """Get a PyTorch dataloader for the specified dataset

    Args:
        mode (str, optional):
            Specify the split of the dataset [training | validation | test]. Defaults to 'training'.
        dataset (str, optional):
            Specify the dataset used [Sintel | Kitti15]. Defaults to 'Sintel'.
        batch_size (int, optional):
            Defaults to 1.
        small_run (bool, optional):
            For debugging: Will load only 32 images. Defaults to False.
        dataset_pass (str, optional):
            Dataset type ['' | clean | final]. Required for Sintel, Things and Driving dataset.

    Raises:
        ValueError: Unknown mode.
        ValueError: Unkown dataset.

    Returns:
        torch.utils.data.DataLoader: Dataloader which can be used for FGSM.
    """

    if dataset == 'Sintel':
        # The Sintel dataset from D.J. Butler et al. "A naturalistic open source movie for optical flow evaluation" (ECCV 2012)
        if mode == 'training':
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"),
                                         root=Paths.config("sintel_mpi"), dataset_pass=dataset_pass, has_gt=True)
        elif mode == 'test':
            # with this option, ground truth and valid are None!!
            dataset = datasets.MpiSintel(split=Paths.splits("sintel_eval"),
                                         root=Paths.config("sintel_mpi"), dataset_pass=dataset_pass, has_gt=False)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == 'Kitti15':
        # The KITTI15 dataset from M. Menze et al. "Object scene flow for autonomous vehicles" (CVPR 2015)
        if mode == 'training':
            dataset = datasets.KITTI(split=Paths.splits(
                "kitti_train"), aug_params=None, root=Paths.config("kitti15"), has_gt=True)
        elif mode == 'test':
            dataset = datasets.KITTI(split=Paths.splits(
                "kitti_eval"), aug_params=None, root=Paths.config("kitti15"), has_gt=False)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "Spring":
        # The Spring dataset from L. Mehl et al. "Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo" (CVPR 2023)
        if mode == 'training':
            dataset = datasets.Spring(split=Paths.splits(
                "spring_train"), root=Paths.config("spring"), has_gt=True, fwd_only=False)
        elif mode == 'test':
            dataset = datasets.Spring(split=Paths.splits(
                "spring_eval"), root=Paths.config("spring"), has_gt=False, fwd_only=False)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "SpringSplitScheurer":
        # Spring-train and validation split from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True,
                                      scenes=["0001", "0004", "0005", "0006", "0007", "0008", "0009", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0020",
                                              "0021", "0022", "0023", "0024", "0025", "0027", "0030", "0033", "0036", "0037", "0038", "0039", "0041", "0043", "0044", "0047"],
                                      fwd_only=True, camera=["left"], half_dimensions=True)
        elif mode == 'validation':
            dataset = datasets.Spring(split=Paths.splits("spring_train"), root=Paths.config("spring"), has_gt=True,
                                      scenes=["0002", "0010", "0018",
                                              "0026", "0032", "0045"],
                                      fwd_only=True, camera=["left"], half_dimensions=True)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "HD1KSplitScheurer":
        # HD1K-train and validation split from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            scenes_tr = ['000000', '000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000010', '000011', '000012', '000014', '000015', '000016',
                         '000017', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029', '000030', '000031', '000033', '000034', '000035']
            dataset = datasets.HD1K(root=Paths.config(
                "hd1k"), has_gt=True, scenes=scenes_tr, half_dimensions=True)
        elif mode == 'validation':
            scenes_val = ["000009", "000013", "000018", "000019", "000032"]
            dataset = datasets.HD1K(root=Paths.config(
                "hd1k"), has_gt=True, scenes=scenes_val, half_dimensions=True)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "DrivingSample":
        # Driving sample from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True,
                                       dataset_pass=[dataset_pass], focallength=["15mm"], drivingcamview=["forward"], direction=["forward"], speed=["fast"], camera=["left"])
        elif mode == 'validation':
            dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True,
                                       dataset_pass=[dataset_pass], focallength=["15mm"], drivingcamview=["forward"], direction=["forward"], speed=["fast"], camera=["left"])
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "Driving":
        # Driving sample from E. Scheurer et al. "Detection defenses: An empty promise against adversarial patch attacks on optical flow " (arXiv 2023)
        if mode == 'training':
            dataset = datasets.Driving(root=Paths.config("driving"), has_gt=True, dataset_pass=[dataset_pass], direction=["forward"])
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "FlyingChairs":
        # The FlyingChairs dataset from P. Fischer et al. "FlowNet: Learning Optical Flow with Convolutional Networks" (arXiv 2015)
        if mode == 'training':
            dataset = datasets.FlyingChairs(split='training')
        elif mode == 'validation':
            dataset = datasets.FlyingChairs(split='validation')
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')

    elif dataset == "FlyingThings3D":
        # The FlyingThings3D dataset from Mayer et al. "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation" (CVPR 2016)
        dataset_pass_folder = f'frames_{dataset_pass}pass'
        if mode == 'training':
            dataset = datasets.FlyingThings3D(
                split_dir='TRAIN', dataset_pass=dataset_pass_folder)
        elif mode == 'validation':
            dataset = datasets.FlyingThings3D(
                split_dir='TEST', dataset_pass=dataset_pass_folder)
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')
    
    elif dataset == 'Viper':
        # The Viper dataset from Richter et al. "Playing for Benchmarks" (ICCV 2017)
        if mode == 'training':
            dataset = datasets.Viper(split=Paths.splits("viper_train"),
                                     root=Paths.config("viper"))
        elif mode == 'validation':
            dataset = datasets.Viper(split=Paths.splits("viper_eval"),
                                     root=Paths.config("viper"))
        else:
            raise ValueError(f'The specified mode: {mode} is unknown.')


    else:
        raise ValueError(
            "Unknown dataset %s, use either 'Sintel', 'Kitti15', 'Spring', 'SintelSplitZhao', 'SpringSplitScheurer', 'HD1KSplitScheurer' or 'DrivingSample'." % (dataset))
    # if e.g. the evaluation dataset does not provide a ground truth this is specified
    ds_has_gt = dataset.has_groundtruth()

    if small_run:
        reduced_num_samples = 32
        rand_indices = np.random.randint(0, len(dataset), reduced_num_samples)
        indices = np.arange(0, reduced_num_samples)
        dataset = Subset(dataset, indices)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), ds_has_gt


def preprocess_img(network, *images):
    """Manipulate input images, such that the specified network is able to handle them

    Args:
        network (str):
            Specify the network to which the input images are adapted

    Returns:
        InputPadder, *tensor:
            returns the Padder object used to adapt the image dimensions as well as the transformed images
    """
    if network in ['RAFT', 'GMA', 'FlowFormer', 'SKFlow']:
        padder = InputPadder(images[0].shape)
        output = padder.pad(*images)

    elif network in ['PWCNet', 'IRR-PWC']:
        images = [(img / 255.) for img in images]
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network == 'SpyNet':
        # normalize images to [0, 1]
        images = [img / 255. for img in images]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network[:7] == 'FlowNet':
        # normalization only for FlowNet, not FlowNet2
        if not network[:8] == 'FlowNet2':
            images = [img / 255. for img in images]
        # make image divisibile by 64
        padder = InputPadder(images[0].shape, divisor=64)
        output = padder.pad(*images)

    elif network[:7] == 'MemFlow':
        images = [2 * (img / 255.0) - 1.0 for img in images]
        padder = InputPadder(images[0].shape)
        output = padder.pad(*images)

    elif network in ['unimatch', 'MS-RAFT+', 'CCMR+'] or network.startswith('unimatch'):
        padder = InputPadder(images[0].shape, divisor=16)
        output = padder.pad(*images)

    else:
        padder = None
        output = images

    return padder, output


def postprocess_flow(network, padder, *flows):
    """Manipulate the output flow by removing the padding

    Args:
        network (str): name of the network used to create the flow
        padder (InputPadder): instance of InputPadder class used during preprocessing
        flows (*tensor): (batch) of flow fields

    Returns:
        *tensor: output with removed padding
    """

    if padder != None:
        # remove padding
        return [padder.unpad(flow) for flow in flows]

    else:
        return flows


def compute_flow(model, network, x1, x2, test_mode=True, **kwargs):
    """subroutine to call the forward pass of the network

    Args:
        model (torch.nn.module):
            instance of optical flow model
        network (str):
            name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
        x1 (tensor):
            first image of a frame sequence
        x2 (tensor):
            second image of a frame sequence
        test_mode (bool, optional):
            applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

    Returns:
        tensor: optical flow field
    """
    if network == "scaled_input_model":
        flow = model(x1, x2, test_mode=True, **kwargs)

    elif network == 'RAFT':
        _, flow = model(x1, x2, test_mode=test_mode, **kwargs)

    elif network == 'GMA':
        _, flow = model(x1, x2, iters=6, test_mode=test_mode, **kwargs)

    elif network == 'SKFlow':
        _, flow = model(x1, x2, iters=15, test_mode=True)

    elif network == 'FlowFormer':
        flow = model(x1, x2)[0]

    elif network == 'FlowNetCRobust':
        flow = model(x1, x2)

    elif network[:7] == 'FlowNet':
        # all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
        x = torch.stack((x1, x2), dim=-3)
        # FlowNet2-variants: all fine now, input [0,255] is taken.

        if not network[:8] == 'FlowNet2':
            # FlowNet variants need input in [-1,1], which is achieved by substracting the mean rgb value from the image in [0,1]
            rgb_mean = x.contiguous().view(
                x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1, 1, 1,)).detach()
            x = x - rgb_mean

        flow = model(x)

    elif network[:7] == 'MemFlow':
        # all MemFlow types need image tensor of dimensions [batch, image12, colors, x, y] = [b,2,3,x,y]
        from models.MemFlow.inference.inference_core_skflow import InferenceCore
        # [ugly hack cont.] read config
        cfg = model.config
        images = torch.stack((x1, x2), dim=-4)

        processor = InferenceCore(model.module, config=cfg)
        ti = 0
        flow_low, flow_pre = processor.step(images[:, ti:ti+2],
                                            end=(ti == images.shape[1]-2),
                                            add_pe=(
                                                'rope' in cfg and cfg.rope),
                                            flow_init=None)
        flow = flow_pre

    elif network in ['PWCNet', 'SpyNet']:
        with warnings.catch_warnings():
            # this will catch the deprecated warning for spynet and pwcnet to avoid messy console
            warnings.filterwarnings(
                "ignore", message="nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
            warnings.filterwarnings(
                "ignore", message="Default upsampling behavior when mode={} is changed")
            warnings.simplefilter("ignore", UserWarning)
            flow = model(x1, x2, **kwargs)

    elif network.startswith('unimatch'):
        # cf. unimatch/scripts/gmflow_evaluate
        kwargs_unimatch = dict(
            attn_type='swin',
            num_reg_refine=1,
            task='flow',
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
        )
        if 'scale2' in network or model.module.num_scales == 2:
            assert model.module.num_scales == 2
            kwargs_unimatch.update(dict(
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1],
                ))
        if model.module.num_scales == 2:
                kwargs_unimatch.update(dict(
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1],
                ))
        if 'regrefine' in network or model.module.reg_refine:
            assert model.module.reg_refine
            kwargs_unimatch.update(dict(
                num_reg_refine=6,
            ))

        results_dict = model(x1, x2, **kwargs_unimatch)
        flow = results_dict['flow_preds'][-1]

    elif network[:8] == 'SEA-RAFT':
        # iters for "model-L" should be 12 ...
        output = model(x1, x2, iters=model.args.iters, test_mode=True)
        flow = output['flow'][-1]

    elif network[:4] == 'IRR-':
        output = model({
            'input1': x1,
            'input2': x2,
        })
        flow = output['flow']

    elif network[:8] == 'ptlflow-':
        from ptlflow.utils.io_adapter import IOAdapter

        x1 = np.array(x1[0].cpu()).transpose((1, 2, 0))
        x2 = np.array(x2[0].cpu()).transpose((1, 2, 0))
        
        # this is preprocessing but needs the model instance
        io_adapter = IOAdapter(model, x1.shape[-2:])
        inputs = io_adapter.prepare_inputs(
            images = [x1, x2],
            image_only=True
        )

        inputs['images'] = inputs['images'].to('cuda') / 255

        predictions = model(inputs)
        predictions = io_adapter.unscale(predictions)
        flows = predictions['flows']
        assert len(flows) == 1
        flow = flows[0]

    elif network == 'MS-RAFT+':
        flow_low, flow = model(
            x1, x2, iters=[4, 6, 5, 10], flow_init=None, test_mode=True)

    elif network == 'CCMR+':
        sintel_iters = [8, 10, 10, 10]
        flow_low, flow = model(x1, x2, iters=sintel_iters, test_mode=True)

    elif network in ['MatchFlowG', 'MatchFlowR']:
        original_shape = x1.shape[2:]
        new_shape = int(
            np.ceil(original_shape[0] / 32) * 32), int(np.ceil(original_shape[1] / 32) * 32)
        x1 = F.interpolate(
            x1, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)
        x2 = F.interpolate(
            x2, (new_shape[0], new_shape[1]), mode='bilinear', align_corners=True)

        _, flow_pr = model(x1, x2, iters=32, test_mode=True)

        flow = F.interpolate(flow_pr, (original_shape[0], original_shape[1]),
                             mode='bilinear', align_corners=True)[0] \
            .cpu() * torch.tensor([original_shape[1] / new_shape[1], original_shape[0] / new_shape[0]]).view(2, 1, 1)

    else:
        flow = model(x1, x2, **kwargs)

    return flow


def model_takes_unit_input(model):
    """Boolean check if a network needs input in range [0,1] or [0,255]

    Args:
        model (str):
            name of the model

    Returns:
        bool: True -> [0,1], False -> [0,255]
    """
    model_takes_unit_input = False
    if model in ["PWCNet", "SpyNet", "FlowNetCRobust"]:
        model_takes_unit_input = True
    return model_takes_unit_input


def get_dimensions(data_loader):
    """Return the image dimensions of the first element of a data_loader

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader of an image dataset

    Returns:
        int, int: image height and image width
    """
    temp_img, _, _, _ = next(iter(data_loader))
    img_h, img_w = temp_img.size()[-2:]
    return img_h, img_w


def flow_length(flow):
    """Calculates the length of the flow vectors of a flow field

    Args:
        flow (tensor):
            flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        torch.float: length of the flow vectors f_ij, computed as sqrt(u_ij^2 + v_ij^2) in a tensor of (b,1,H,W) or (1,H,W)
    """
    flow_pow = torch.pow(flow, 2)
    flow_norm_pow = torch.sum(flow_pow, -3, keepdim=True)

    return torch.sqrt(flow_norm_pow)


def maximum_flow(flow):
    """Calculates the length of the longest flow vector of a flow field

    Args:
        flow (tensor):
            a flow field tensor of dimensions (b,2,H,W) or (2,H,W)

    Returns:
        float: length of the longest flow vector f_ij, computed as sqrt(u_ij^2 + v_ij^2)
    """
    return torch.max(flow_length(flow)).cpu().detach().numpy()


def quickvis_tensor(t, filename):
    """Saves a tensor with three dimensions as image to a specified file location.

    Args:
        t (tensor):
            3-dimensional tensor, following the dimension order (c,H,W)
        filename (str):
            name for the image to save, including path and file extension
    """
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(t.size()) == 3:
        img = t.detach().cpu().numpy()
        valid = True

    elif len(t.size()) == 4 and t.size()[0] == 1:
        img = t[0, :, :, :].detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %
              str(t.size()))

    if valid:
        img = np.rollaxis(img, 0, 3)
        data = img.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_tensor(t, filename, min=0., max=255.):
    """Saves a batch (>= 1) of image tensors with three dimensions as images to a specified file location.
    Also rescales the color values according to the specified range of the color scale.

    Args:
        t (tensor):
            batch of 3-dimensional tensor, following the dimension order (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension. Batches will append a number at the end of the filename.
        min (float, optional):
            minimum value of the color scale used by tensor. Defaults to 0.
        max (float, optional):
            maximum value of the color scale used by tensor Defaults to 255.
    """
    # rescale to [0,255]
    t = (t.detach().clone() - min) / (max - min) * 255.

    if len(t.size()) == 3 or (len(t.size()) == 4 and t.size()[0] == 1):
        quickvis_tensor(t, filename)

    elif len(t.size()) == 4:
        for i in range(t.size()[0]):
            if i == 0:
                quickvis_tensor(t[i, :, :, :], filename)
            else:
                quickvis_tensor(t[i, :, :, :], filename+"_"+str(i))

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." % str(
            t.size()))


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a flow field tensor with two dimensions as image to a specified file location.

    Args:
        flow (tensor):
            2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    # check if filename already contains .png extension
    if not filename.endswith('.png'):
        filename += '.png'
    valid = False
    if len(flow.size()) == 3:
        flow_img = flow.clone().detach().cpu().numpy()
        valid = True

    elif len(flow.size()) == 4 and flow.size()[0] == 1:
        flow_img = flow[0, :, :, :].clone().detach().cpu().numpy()
        valid = True

    else:
        print("Encountered invalid tensor dimensions %s, abort printing." %
              str(flow.size()))

    if valid:
        # make directory and ignore if it exists
        if not os.path.dirname(filename) == "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        # write flow
        flow_img = np.rollaxis(flow_img, 0, 3)
        data = colorplot_light(
            flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False)
        data = data.astype(np.uint8)
        data = Image.fromarray(data)
        data.save(filename)


def quickvisualization_flow(flow, filename, auto_scale=True, max_scale=-1):
    """Saves a batch (>= 1) of 2-dimensional flow field tensors as images to a specified file location.

    Args:
        flow (tensor):
            single or batch of 2-dimensional flow tensors, following the dimension order (c,H,W) or (b,c,H,W)
        filename (str):
            name for the image to save, including path and file extension.
        auto_scale (bool, optional):
            automatically scale color values. Defaults to True.
        max_scale (int, optional):
            if auto_scale is false, scale flow by this value. Defaults to -1.
    """
    if len(flow.size()) == 3 or (len(flow.size()) == 4 and flow.size()[0] == 1):
        quickvis_flow(flow, filename, auto_scale=auto_scale,
                      max_scale=max_scale)

    elif len(flow.size()) == 4:
        for i in range(flow.size()[0]):
            if i == 0:
                quickvis_flow(flow[i, :, :, :], filename,
                              auto_scale=auto_scale, max_scale=max_scale)
            else:
                quickvis_flow(flow[i, :, :, :], filename+"_"+str(i),
                              auto_scale=auto_scale, max_scale=max_scale)

    else:
        print("Encountered unprocessable tensor dimensions %s, abort printing." % str(
            flow.size()))


def torchfloat_to_float64(torch_float):
    """helper function to convert a torch.float to numpy float

    Args:
        torch_float (torch.float):
            scalar floating point number in torch

    Returns:
        numpy.float: floating point number in numpy
    """
    float_val = float(torch_float.detach().cpu().numpy())
    return float_val


def args_to_outputfilepath(prefix, net, weight_path, dataset, dataset_stage, dataset_pass, top_folder='accuracy', filename='metrics.json'):
    # IRR weights are stored in a folder hierarchy
    if net in ['IRR-PWC']:
        if weight_path.endswith('_best.ckpt'):
            weight_name = str(weight_path).split('/')[-2] + '_best'
        elif weight_path.endswith('_latest.ckpt'):
            weight_name = str(weight_path).split('/')[-2] + '_latest'
        else:
            weight_name = str(weight_path).split('/')[-2]
    else:
        weight_name = weight_path.split("/")[-1]
    
    return os.path.join(
        prefix,
        top_folder,
        dataset,
        dataset_pass,
        dataset_stage,
        net,
        weight_name,
        filename
    )
