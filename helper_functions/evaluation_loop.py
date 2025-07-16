import os
import numpy as np
from tqdm import tqdm

from helper_functions.config_specs import ProgBar
from helper_functions import ownutilities

try:
    import matplotlib.pyplot as plt
    from flow_library.flow_show import getFlowVis
except ImportError as e:
    print('will not save plots', e)
    plt = None

# pre-compute weights as constants
__wi = 1 - np.arange(0, 100, dtype=float) / 100.
__sum_wi = np.sum(__wi)
__deltai = np.arange(1, 101, dtype=float) / 20.0
def get_wauc(epe):
    # cf https://github.com/cv-stuttgart/springwebsite/blob/main/springeval/management/commands/evaluation.py#L86
    N = (~np.isnan(epe)).sum()

    _err = (epe[..., None] <= __deltai).sum(axis=0)
    _wauc = np.sum(__wi * _err)
    _wauc = _wauc / (N * __sum_wi)
    wauc = _wauc

    # wauc = 0
    # sum_wi = 0
    # for i in range(1, 101):
    #     wi = 1 - ((i-1) / 100.0)
    #     deltai = i / 20.0
    #     err = (epe <= deltai).sum()
    #     wauc += wi * err
    #     sum_wi += wi
    # wauc = wauc / (N*sum_wi)
    # # assert wauc == _wauc, (wauc, _wauc)
    # assert np.allclose(wauc, _wauc), (wauc, _wauc)
    return wauc


def eval_metrics_flat(flow_pr_flat, flow_gt_flat):
    ''' calculate various metrics

    The calculation of the outlier pixels needed for the
    kitti Fl-all score is adopted from the RAFT repository.
    (cf. <https://github.com/princeton-vl/RAFT/blob/3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02/evaluate.py#L148>)

    Parameters:

    flow_pr_flat: np.ndarray 2xN - the predicted flow for N pixels
    flow_gt_flat: np.ndarray 2xN - the ground truth flow
    '''
    # magnitude of the ground truth flow in each pixel
    mag = np.sqrt(np.sum(flow_gt_flat**2, axis=0))
    mag = np.where(np.isclose(mag, 0), 1e-8, mag)  # avoid division by zero
    _sum = np.sum((flow_pr_flat - flow_gt_flat)**2, axis=0)
    epe_flat = np.sqrt(_sum)

    # correct_pixels = ((epe_flat < 3.0) | (epe_flat/mag < 0.05)))
    bad_pixels = ((epe_flat >= 3.0) & ((epe_flat/mag) >= 0.05))

    return {
        'aee': epe_flat.mean(),
        'fl-epe-1px': np.mean(epe_flat < 1, dtype=float),
        'fl-epe-3px': np.mean(epe_flat < 3, dtype=float),
        'fl-epe-5px': np.mean(epe_flat < 5, dtype=float),
        'fl-all-kitti': np.mean(bad_pixels.astype(float)),
        'mse': np.mean((flow_pr_flat - flow_gt_flat)**2),
        'wauc': get_wauc(epe_flat),
        'num_valid_pixels': epe_flat.shape[0],
    }


def _combine_metrics(res_list):
    '''Combine metrics as weighted mean, weighted by the number of valid pixels.

    Except for:
    'num_valid_pixels' - total number of valid pixels, i.e., sum of the list_item['num_valid_pixels']
    '''
    num_pixels = sum(x['num_valid_pixels'] for x in res_list)

    res = {
        # compute overall metrics as weighted sums
        metric_key: sum(x[metric_key] * x['num_valid_pixels']
                        for x in res_list) / num_pixels
        for metric_key in res_list[0].keys() if metric_key != 'num_valid_pixels'
    }
    res['num_valid_pixels'] = num_pixels
    return res


def evaluate_model_on_dataset(model, dataset, net: str, device, callback_frame_result=None):
    print('Evaluate on ', len(dataset), 'samples')
    intermediate_results = []
    prog_bar_dataset = tqdm(total=len(dataset), desc='Image', bar_format=ProgBar.settings(
        'format_eval'), disable=ProgBar.settings('disable'))

    for test_id in range(0, len(dataset)):
        img1, img2, flow_gt, valid = dataset[test_id]

        image1, image2 = img1[None].to(device), img2[None].to(device)
        padder, [image1, image2] = ownutilities.preprocess_img(
            net, image1, image2)

        # calculate attacked flow
        flow_pr = ownutilities.compute_flow(
            model, net, image1, image2)
        [flow_pr] = ownutilities.postprocess_flow(
            net, padder, flow_pr)

        # evaluate on cpu
        flow_pr = flow_pr.cpu().numpy()
        flow_gt = flow_gt.cpu().numpy()
        valid = valid.cpu().numpy() >= 0.5

        assert valid.shape == flow_gt.shape[-2:], (flow_gt.shape,
                                                   valid.shape, flow_pr.shape)
        assert flow_gt.reshape(-1).shape == flow_pr.reshape(-1).shape, \
            (flow_gt.shape, flow_pr.shape)
        assert np.all((valid == 0.) | (valid == 1.)), valid.unique()

        # select valid pixels
        flow_pr_flat = flow_pr.reshape(2, -1)[:, valid.reshape(-1)]
        flow_gt_flat = flow_gt.reshape(2, -1)[:, valid.reshape(-1)]

        intermediate_results.append(
            eval_metrics_flat(flow_pr_flat, flow_gt_flat))
        prog_bar_dataset.update()

        if callback_frame_result:
            callback_frame_result(test_id, intermediate_results[-1])
        
        if plt and False:
            path = f'output/sample_flows/{type(dataset).__name__}'
            if not os.path.exists(path):
                os.makedirs(path)
            plt.imshow(img1.numpy().transpose((1, 2, 0)) / 255)
            plt.axis('off')
            plt.savefig(f'{path}/{test_id}_img1.png')
            plt.close()
            plt.imshow(img2.numpy().transpose((1, 2, 0)) / 255)
            plt.axis('off')
            plt.savefig(f'{path}/{test_id}_img2.png')
            plt.close()
            rgb_vis_gt, max_scale_gt = getFlowVis(flow_gt.transpose((1, 2, 0)), auto_scale=True, return_max=True)
            rgb_vis_pr, max_scale_pr = getFlowVis(flow_pr[0].transpose((1, 2, 0)), auto_scale=True, return_max=True)

            if max_scale_pr > max_scale_gt:
                mult_pr = max_scale_gt / max_scale_pr
                rgb_vis_pr = rgb_vis_pr * mult_pr
                rgb_vis_gt = rgb_vis_gt / 255
                rgb_vis_pr = rgb_vis_pr / 255
            else:
                mult_gt = max_scale_pr / max_scale_gt
                rgb_vis_gt = rgb_vis_gt * mult_gt
                rgb_vis_gt = rgb_vis_gt / 255
                rgb_vis_pr = rgb_vis_pr / 255


            print('Ground Truth', max_scale_gt)
            plt.imshow(rgb_vis_gt)
            plt.axis('off')
            plt.savefig(f'{path}/{test_id}_flow_gt.png')
            plt.close()
            print('Prediction by raft', max_scale_pr)
            plt.imshow(rgb_vis_pr)
            plt.axis('off')
            plt.savefig(f'{path}/{test_id}_flow_pred_{net}.png')
            plt.close()
            rgb_vis, max_scale = getFlowVis((flow_gt - flow_pr[0]).transpose((1, 2, 0)), auto_scale=True, return_max=True)
            print('Difference GT-PR', max_scale)
            plt.imshow(rgb_vis)
            plt.axis('off')
            plt.close()


    global_metrics = _combine_metrics(intermediate_results)
    return intermediate_results, global_metrics
