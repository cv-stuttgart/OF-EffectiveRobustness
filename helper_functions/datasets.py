# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import re
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import cv2

from helper_functions import frame_utils
from helper_functions.config_specs import Paths, Conf


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.sparse = sparse

        self.has_gt = False
        self.init_seed = False
        # only required for Spring, because its ground truth is 2x the image size in every dimension
        self.subsample_groundtruth = False
        self.half_dimensions = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.enforce_dimensions = False
        self.image_x_dim = 0
        self.image_y_dim = 0

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        if self.half_dimensions:
            img1 = cv2.resize(
                img1, (int(img1.shape[1]/2.), int(img1.shape[0]/2.)))
            img2 = cv2.resize(
                img2, (int(img2.shape[1]/2.), int(img2.shape[0]/2.)))
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        valid = None

        if self.has_gt:
            if self.sparse:
                flow, valid = frame_utils.read_flow_sparse(self.flow_list[index])
                if self.half_dimensions:
                    valid = valid[::2, ::2]
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
                if self.subsample_groundtruth:
                    # use only every second value in both spatial directions ==> flow will have same dimensions as images for Spring
                    flow = flow[::2, ::2]
            flow = np.array(flow).astype(np.float32)
            if self.half_dimensions:
                flow = flow[::2, ::2]
                # flow = cv2.resize(flow, (int(flow.shape[1]/2.), int(flow.shape[0]/2.)))
                flow = flow / 2.

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        else:
            (img_x, img_y, img_chann) = img1.shape

            # make correct size for flow (2 dimensions for u,v instead of 3 [r,g,b] for image )
            flow = np.zeros((img_x, img_y, 2))
            valid = False

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if self.enforce_dimensions:
            dims = img1.size()
            x_dims = dims[-2]
            y_dims = dims[-1]

            diff_x = self.image_x_dim - x_dims
            diff_y = self.image_y_dim - y_dims

            img1 = F.pad(img1, (0, diff_y, 0, diff_x), "constant", 0)
            img2 = F.pad(img2, (0, diff_y, 0, diff_x), "constant", 0)

            flow = F.pad(flow, (0, diff_y, 0, diff_x), "constant", 0)
            if self.has_gt:
                valid = F.pad(valid, (0, diff_y, 0, diff_x), "constant", False)

        return img1, img2, flow, valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

    def has_groundtruth(self):
        return self.has_gt


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split=Paths.splits("sintel_train"), root=Paths.config("sintel_mpi"), dataset_pass='clean', has_gt=False, scenes=None, every_nth_img=1):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dataset_pass)

        self.has_gt = has_gt

        for scene in sorted(os.listdir(image_root)):
            if scenes is None or scene in scenes:
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    self.image_list += [[image_list[i], image_list[i+1]]]
                    self.extra_info += [(scene, i)]  # scene and frame_id

                if not split == Paths.splits("sintel_eval"):
                    self.flow_list += sorted(
                        glob(osp.join(flow_root, scene, '*.flo')))

        if every_nth_img > 1:
            self.image_list = self.image_list[::every_nth_img]
            self.extra_info = self.extra_info[::every_nth_img]
            self.flow_list = self.flow_list[::every_nth_img]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the MPI Sintel dataset." % root)


class Spring(FlowDataset):
    def __init__(self, aug_params=None, split=Paths.splits("spring_eval"), root=Paths.config("spring"), has_gt=False, scenes=None, subsample_groundtruth=True, fwd_only=False, camera=["left", "right"], half_dimensions=False):
        super(Spring, self).__init__(aug_params)
        image_root = os.path.join(root, split)

        self.has_gt = has_gt
        self.subsample_groundtruth = subsample_groundtruth
        self.half_dimensions = half_dimensions

        for scene in sorted(os.listdir(image_root)):
            if scenes is None or scene in scenes:
                for cam in camera:
                    image_list = sorted(
                        glob(os.path.join(image_root, scene, f"frame_{cam}", '*.png')))
                    # forward
                    for i in range(len(image_list)-1):
                        self.image_list += [[image_list[i], image_list[i+1]]]
                        self.extra_info += [(scene, i+1, cam, "FW")]
                    # backward
                    if not fwd_only:
                        # for i in reversed(range(1, len(image_list))):
                        for i in range(1, len(image_list)):
                            self.image_list += [[image_list[i],
                                                 image_list[i-1]]]
                            self.extra_info += [(scene, i+1, cam, "BW")]

                    if split != 'test':
                        self.flow_list += sorted(
                            glob(osp.join(image_root, scene, f"flow_FW_{cam}", '*.flo5')))
                        if not fwd_only:
                            self.flow_list += sorted(
                                glob(osp.join(image_root, scene, f"flow_BW_{cam}", '*.flo5')))
                        pass

        # # Quick fix for broken file Spring/train/0018/flow_FW_left/flow_FW_left_0014.flo5 on haendel
        dubious_file_idx = next((i for i, x in enumerate(self.flow_list) if x.endswith('train/0018/flow_FW_left/flow_FW_left_0014.flo5')),
                                -1)
        if Conf.config('hostname') == 'haendel' and dubious_file_idx > 0:
            self.flow_list[dubious_file_idx] = '/data/katrin/datasets/Spring/train/0018/flow_FW_left/flow_FW_left_0014.flo5'
            print('Spring Dataset: Replaced path to /data/katrin/datasets/Spring/train/0018/flow_FW_left/flow_FW_left_0014.flo5')
        # else:
        #     assert all(not x.endswith('train/0018/flow_FW_left/flow_FW_left_0014.flo5') for x in self.flow_list)

        assert len(self.image_list) == len(self.flow_list)
        if len(self.image_list) == 0:
            raise RuntimeWarning(
                f"No Spring data found at dataset root '{root}'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the Spring dataset.")


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split=Paths.splits("kitti_train"), root=Paths.config("kitti15"), has_gt=False):
        super(KITTI, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the KITTI dataset." % root)


class Driving(FlowDataset):
    def __init__(self, aug_params=None, root=Paths.config("driving"), has_gt=True, dataset_pass=['clean', 'final'], focallength=["15mm", "30mm"], drivingcamview=["forward", "backward"], direction=["forward", "backward"], speed=["fast", "slow"], camera=["left", "right"]):
        super(Driving, self).__init__(aug_params)

        self.has_gt = has_gt

        for ds in dataset_pass:
            for fl in focallength:
                for dr in drivingcamview:
                    for sp in speed:
                        for cm in camera:

                            images = sorted(glob(osp.join(
                                root, f"frames_{ds}pass", f"{fl}_focallength", f"scene_{dr}s", sp, cm, '*.png')))

                            if "forward" in direction:
                                for img1, img2 in zip(images[:-1], images[1:]):
                                    frame_id = img1.split('/')[-1]
                                    self.extra_info += [
                                        [frame_id, f"frames_{ds}pass", f"{fl}_focallength", f"scene_{dr}s", sp, cm, "FW"]]
                                    self.image_list += [[img1, img2]]
                                if self.has_gt:
                                    self.flow_list += sorted(glob(osp.join(
                                        root, "optical_flow", f"{fl}_focallength", f"scene_{dr}s", sp, "into_future", cm, 'OpticalFlowIntoFuture*.pfm')))[:-1]

                            if "backward" in direction:
                                for img1, img2 in zip(images[1:], images[:-1]):
                                    frame_id = img1.split('/')[-1]
                                    self.extra_info += [
                                        [frame_id, f"frames_{ds}pass", f"{fl}_focallength", f"scene_{dr}s", sp, cm, "BW"]]
                                    self.image_list += [[img1, img2]]
                                if self.has_gt:
                                    self.flow_list = sorted(glob(osp.join(
                                        root, "optical_flow", f"{fl}_focallength", f"scene_{dr}s", sp, "into_past", cm, 'OpticalFlowIntoPast*.pfm')))[1:]


        
        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No Driving data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the Driving dataset." % root)
        assert len(self.image_list) == len(self.flow_list), (len(self.image_list), len(self.flow_list), has_gt)


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root=Paths.config("hd1k"), has_gt=True, every_nth_img=1, half_dimensions=False, scenes=None):
        super(HD1K, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt
        self.half_dimensions = half_dimensions

        scene_ids = np.unique(np.array([pth.split("/")[-1].split("_")[0] for pth in sorted(
            glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/*.png')))]))

        for scene in scene_ids:
            if scenes is None or scene in scenes:
                flows = sorted(
                    glob(os.path.join(root, 'hd1k_flow_gt', f'flow_occ/{scene}_*.png')))
                images = sorted(
                    glob(os.path.join(root, 'hd1k_input', f'image_2/{scene}_*.png')))

                for i in range(len(flows)-1):
                    self.flow_list += [flows[i]]
                    self.image_list += [[images[i], images[i+1]]]
                    seqid = images[i].split(
                        "/")[-1].split("_")[-1].split(".")[0]
                    self.extra_info += [(scene, seqid)]

        # images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/*.png')))

        # seq_ix = 0
        # while 1:
        #     flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
        #     images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

        #     if len(flows) == 0:
        #         break

        #     for i in range(len(flows)-1):
        #         self.flow_list += [flows[i]]
        #         self.image_list += [ [images[i], images[i+1]] ]

        #     seq_ix += 1

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No HD1K data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the HD1K dataset." % root)

        if every_nth_img > 1:
            self.image_list = self.image_list[::every_nth_img]
            self.extra_info = self.extra_info[::every_nth_img]
            self.flow_list = self.flow_list[::every_nth_img]


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=Paths.config("flying_chairs"), split_path=f'./FlyingChairs_train_val.txt'):
        super(FlyingChairs, self).__init__(aug_params)

        self.split_path = split_path
        self.has_gt = True

        images = sorted(glob(osp.join(root, 'data', '*.ppm')))
        flows = sorted(glob(osp.join(root, 'data', '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt(split_path, dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No FlyingChairs data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the FlyingChairs dataset." % root)


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, split_dir='TRAIN', root=Paths.config("flying_things"), dataset_pass='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        self.has_gt = True

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(
                    glob(osp.join(root, dataset_pass, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(
                    glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam)
                                   for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        elif direction == 'into_past':
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No FlyingThings3D data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the FlyingThings3D dataset." % root)


class Viper(FlowDataset):
    def __init__(self, aug_params=None, root=Paths.config("viper"), split='train'):
        super(Viper, self).__init__(aug_params, sparse=True)

        self.has_gt = True

        root = osp.join(root, split)
        img_root = osp.join(root, 'img')
        flow_root = osp.join(root, 'flow')

        if split == 'train':            # ignore large outliers
            self.max_flow = 2048

        flows = []
        imgs = []
        info = []

        for group in sorted(os.listdir(img_root)):
            for img1 in sorted(os.listdir(osp.join(img_root, group))):
                num = int(re.match(group + "_([0-9]*).jpg", img1).group(1))

                img1 = osp.join(img_root, group, img1)
                img2 = osp.join(img_root, group, f"{group}_{num + 1:05d}.jpg")
                flow = osp.join(flow_root, group, f"{group}_{num:05d}.npz")

                if not osp.exists(img2):
                    continue

                if osp.getsize(img1) == 0 or osp.getsize(img2) == 0:
                    print('[Viper] issue with image file', img1, img2)
                    continue

                if split == 'test':
                    flow = None
                else:
                    if not osp.exists(flow) or osp.getsize(flow) == 0:
                        print('[Viper] flow missing', flow)
                        continue

                imgs += [(img1, img2)]
                flows += [flow]
                info += [(group, num)]

        self.image_list = imgs
        self.flow_list = flows
        self.extra_info = info

        if len(self.image_list) == 0:
            raise RuntimeWarning(
                "No Viper data found at dataset root '%s'. Check the configuration file under helper_functions/config_specs.py and add the correct path to the Viper dataset." % root)
