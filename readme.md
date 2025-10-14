# Effective Robustness for Optical Flow

This repository contains the source code for:

> **[On the Generalization of Optical Flow: Quantifying Robustness to Dataset Shifts](https://openreview.net/pdf/1f9803811ff2cbef605e965b3c76d2fc259e0630.pdf)** <br/>
> _ICCV 2025 Workshop DataCV_ <br/>
> Katrin Bauer, Andr&#233;s Bruhn, and Jenny Schmalfuss


## Abstract

Optical flow models are commonly evaluated by their ability to accurately predict the apparent motion from image sequence data.
Though not seen during training, this evaluation data generally shares the training data's characteristics because it stems from the same distribution, i.e., it is in-distribution (ID) with the training data.
However, when models are applied in the real world, the test data characteristics may be shifted, i.e. out-of-distribution (OOD), compared to the training data.
For optical flow models, the generalization to dataset shifts is much less reported than the typical accuracy on ID data.
In this work, we close this gap and systematically investigate the generalization of optical flow models by disentangling accuracy and robustness to dataset shifts with a new effective robustness metric.
We evaluate a testbed of 20 models on six established optical flow datasets.
Across models and datasets, we find that ID accuracy can be used as a predictor for OOD performance, but certain models generalize better than this trend suggests.
While our analysis reveals that model generalization capabilities declined in recent years, we also find that more training data and smart architectural choices can improve generalization.
Across tested models, effective robustness to dataset shifts is high for models that avoid attention mechanisms and favor multi-scale designs.

## Initial setup

### Cloning the repo

Clone this repository and all its submodules:

```shell
git clone thisrepository
git submodule update --init --recursive --remote
```

### Setup virtual environment

```shell
python3 -m venv effrob
source effrob/bin/activate
python -m pip install --upgrade pip
```

### Install required packages

Change into scripts_setup folder and execute the script which installs all required packages via pip.
As each package is installed succesively, you can debug errors for specific packages later.

```shell

cd scripts_setup
bash install_packages.sh
cd ..
```

Depending on your CUDA version, you may need to install an older PyTorch version.
All models except `ptlflow` were tested with

```sh
# Install PyTorch. The torch version depends on your CUDA version.
# For available versions, see https://pytorch.org/get-started/previous-versions/.
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.26.4
```

Using `ptlflow` requires PyTorch version 2.
Checkout [`models/ptlflow/requirements.txt`](models/ptlflow/requirements.txt) for the respective requirements.

<!--
Now ['RAFT', 'GMA', 'FlowFormer', 'FlowNet2S', 'IRR-PWC', 'MemFlow', 'SKFlow', 'unimatch-scale2-regrefine6', 'unimatch-scale2', 'unimatch-scale1', 'SEA-RAFT-S', 'SEA-RAFT-M', 'SEA-RAFT-L'] should work.
Failing:
- 'FlowNet2', 'FlowNet2C' missing CUDA extensions
- 'ptlflow-rpknet' needs torch>=2, CUDA>=12
- 'MatchFlowR', 'MatchFlowG' missing QuadTreeAttention
- 'MS-RAFT+', 'CCMR+' missing alt_cuda_corr
-->

#### Alt. CUDA correlation

The module `alt_cuda_corr` can reduce memory load and is required for MS-RAFT+ and CCMR+.

```shell
cd models/ptlflow/ptlflow/utils/external/alt_cuda_corr/
python setup.py install
```

#### Quad Tree Attention for MatchFlow

```shell
cd models/MatchFlow/QuadTreeAttention/
python setup.py install
```

#### Compiling Cuda Extensions for FlowNet2

Please refer to the [pytorch documentation](https://github.com/NVIDIA/flownet2-pytorch.git) how to compile the channelnorm, correlation and resample2d extensions.
If all else fails, go to the extension folders `/models/FlowNet/{channelnorm,correlation,resample2d}_package`, manually execute

```shell
python3 setup.py install
```

and potentially replace `cxx_args = ['-std=c++11']` by `cxx_args = ['-std=c++14']`, and the list of `nvcc_args` by `nvcc_args = []` in every setup.py file.
If manually compiling worked, you may need to add the paths to the respective .egg files in the `{channelnorm,correlation,resample2d}.py` files, e.g. for channelnorm via

```python
sys.path.append("/lib/pythonX.X/site-packages/channelnorm_cuda-0.0.0-py3.6-linux-x86_64.egg")
import channelnorm_cuda
```

The `site-packages` folder location varies depending on your operation system and python version.

#### Spatial Correlation Sampler

If the installation of the `spatial-correlation-sampler` works and you have a cuda capable machine, open `helper_functions/config_specs.py` and make sure to set the variable `"correlationSamplerOnlyCPU":` to `False`. This will speed up computations when using PWCNet.

If the spatial-correlation-sampler does not install run the following script to install a cpu-only version:

```shell
cd scripts
bash install_scs_cpu.sh
```

When loading gcc and CUDA versions from modules, you need to make sure the versions are compatible and may adjust `GCC_HOME` and other variables. See more informations in [this issue](https://github.com/ClementPinard/Pytorch-Correlation-extension/issues/1#issuecomment-478052992). One solution presented changes the variables as follows:
<details>

``` bash
## used to compile .cu and for cudnn 
export GCC_HOME=/path/to/gcc-5.4.0/
export PATH=$GCC_HOME/bin/:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib:$GCC_HOME/lib64:$GCC_HOME/libexec:$LD_LIBRARY_PATH
export CPLUS_INCLUDE="$GCC_HOME/include:$CPLUS_INCLUDE"
export C_INCLUDE="$GCC_HOME/include:$C_INCLUDE"
export CXX=$GCC_HOME/bin/g++
export CC=$GCC_HOME/bin/gcc ## for make
CC=$GCC_HOME/bin/gcc ## for cmake

## complime using nvcc with gcc
export EXTRA_NVCCFLAGS="-Xcompiler -std=c++98"

## pip install 
pip install spatial-correlation-sampler
```

</details>

### Datasets

For evaluation, we use the datasets [*FlyingThings3D*](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [*Sintel*](http://sintel.is.tue.mpg.de/), [*KITTI 2015*](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow), [*HD1K*](http://hci-benchmark.iwr.uni-heidelberg.de/), [*Driving*](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [*VIPER*](https://playing-for-benchmarks.org/overview/), [*Spring*](https://spring-benchmark.org/).
The datasets are assumed to be in a similar layout as for training [RAFT](https://github.com/princeton-vl/RAFT#required-data):

```
├── datasets
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
    ├── HD1k
        ├── hd1k_challenge
        ├── hd1k_input
        ├── hd1k_flow_uncertainty
        ├── hd1k_flow_gt
    ├── driving
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── Viper
        ├── val
    ├── Spring
        ├── train
```

If you have them already saved somewhere else, you may link to the files with

```bash
mkdir datasets
cd datasets
ln -s /path/to/Sintel Sintel
ln -s /path/to/KITTI KITTI
```

or specify the paths and names directly in `helper_functions/config_specs.py`.


## Code Usage

### Evaluation of a Model

Evaluating the effective robustness of an optical flow model is done in two steps:

1. Evaluate the model on the in-distribution and out-of-distribution datasets.
   (See [below](#adding-external-models) how to add an external model to this repository.)

```sh
python evaluate_accuracy.py \
    --net YourModel --custom_weight_path path/to/your/ckpt.pth \
    --dataset FlyingThings3D --dataset_stage validation --dataset_pass final

python evaluate_accuracy.py \
    --net YourModel --custom_weight_path path/to/your/ckpt.pth \
    --dataset Kitti15 --dataset_stage training --dataset_pass ""
```

2. Evaluate the effective robustness as difference to the baseline.
   The baselines from the paper are in `paper_results/baseline_{$ID_DATASET}_{$OOD_DATASET}.json`.

```sh
python evaluate_effectiverobustness.py \
    --net YourModel --custom_weight_path path/to/your/ckpt.pth \
    --id_dataset FlyingThings3D --id_dataset_stage validation --id_dataset_pass final \
    --ood_dataset Kitti15 --ood_dataset_stage training --ood_dataset_pass "" \
    --baseline_file paper_results/baseline_FlyingThings3D_Kitti15.json
```

To evaluate a model on all datasets used for the paper, you can use the scripts [evaluate_model_accuracy.sh](scripts_evaluate/evaluate_model_accuracy.sh) and [evaluate_model_effective_robustness.sh](scripts_evaluate/evaluate_model_effective_robustness.sh) in scripts_evalaute/.

1. Adapt the environment variable `NET` and `WPATH` in each file.
2. Run them:
   
```sh
sh scripts_evaluate/evaluate_model_accuracy.sh
sh scripts_evaluate/evaluate_model_effective_robustness.sh
```

#### ... or just provide the WAUC

The above process assumes you're using our scripts to evaluate the accuracy of a model using our folder structure of storing intermediate results.
If you use your own evaluation scripts to compute the WAUC, you can evaluate the effective robustness via:

```sh
python evaluate_effectiverobustness_wauc.py --id-wauc 0.63 --ood-wauc 0.52 --baseline_file paper_results/baseline_FlyingThings3D_Kitti15.json
```

### Fitting a Baseline

We provide the baselines used in the paper in `paper_results/baseline_{$ID_DATASET}_{$OOD_DATASET}.json`.

If you want to fit a baseline to your own set of models, checkout the script [scripts_evaluate/fit_baselines.sh](scripts_evaluate/fit_baseline.sh).


## Data Logging

`evaluate_accuracy.py` stores the accuracy for each model at the following path

```sh
./experiment_data/accuracy/{$DATASET_NAME}/{$DATASET_PASS}/{$DATASET_STAGE}/{$NET}/{$WEIGHT_NAME}/metrics.json
```

`fit_baselines.py` stores the baseline parameters as `experiment_data/baseline_{$ID_DATASET}_{$OOD_DATASET}.json`.

The model accuracies and baselines from the paper are stored in [paper_results](paper_results/).

## Adding External Models

The framework is built such that custom (PyTorch) models can be included.
You can add a model either by adding it to the framework `ptlflow` or directly into this framework.

### Adding a model from ptlflow

This framework contains ptlflow as submodule in models/ptlflow. Any model included in ptlflow should also work here after adding it to the valid program arguments.

1. Follow their excellent documentation on [how to add a model](https://ptlflow.readthedocs.io/en/latest/custom/new_model.html).

2. In [`helper_functions/parsing_file.py`](helper_functions/parsing_file.py#L17), add the model to the possible choices for `--net` using the naming scheme `ptlflow-yourmodelname`.

### Adding an external model directly

To add an own model, perform the following steps:

1. Create a directory `models/your_model` containing all the required files for the model.

2. Make sure that all import calls are updated to the correct folder. I.e change:

    ```python
    from your_utils import your_functions ## old

    ## should be changed to:
    from models.your_model.your_utils import your_functions ## new
    ```

3. In [`helper_functions/ownutilities.py`](helper_functions/ownutilities.py) modify the following functions:
    - [`import_and_load()`](helper_functions/ownutilities.py#L64): Add the following lines:

        ```python
        elif net == 'your_model':
            ## mandatory: import your model i.e:
            from models.your_model import your_model

            ## optional: you can outsource the configuration of your model e.g. as a .json file in models/_config/
            with open("models/_config/your_model_config.json") as file:
                config = json.load(file)
            ## mandatory: initialize model with your_model and load pretrained weights
            model = your_model(config)
            weights = torch.load(path_weights, map_location=device)
            model = load_state_dict(weights)
        ```

    - [`preprocess_img()`](helper_functions/ownutilities.py#L279): Make sure that the input is adapted to the forward pass of your model.
    The dataloader provides rgb images with range `[0, 255]`. The image dimensions differ with the dataset.
    You can use the padder class make the spatial dimensions divisible by a certain divisor.

        ```python
        elif network == 'your_model':
            ## example: normalize rgb range to [0,1]
            images = [(img / 255.) for img in images]
            ## example: initialize padder to make spatial dimension divisible by 64
            padder = InputPadder(images[0].shape, divisor=64)
            ## example: apply padding
            output = padder.pad(*images)
        ```

    - [`model_takes_unit_input()`](helper_functions/ownutilities.py#L392): Add your model to the respective list, if it expects input images in `[0,1]` rather than `[0,255]`.

    - [`compute_flow()`](helper_functions/ownutilities.py#L341): Has to return a tensor `flow` originating from the forward pass of your model with the input images `x1` and `x2`.

    If your model needs further preprocessing like concatenation perform it here:
        ```python
        elif network == 'your_model':
            ## optional: 
            model_input = torch.cat((x1, x2), dim=0)
            ## mandatory: perform forward pass
            flow = model(model_input)
        ```

    - `postprocess_flow()`: Rescale the spatial dimension of the output `flow`, such that they coincide with the original image dimensions. If you used the padder class during preprocessing it will be automatically reused here.

4. Add your model to the possible choices for `--net` in [`helper_functions/parsing_file.py`](helper_functions/parsing_file.py#L17) (i.e. `[... | your_model]`)

## External Models and Dependencies

### Models

The model implementations in models/ are copied from the respective repositories:

- [*RAFT*](https://github.com/princeton-vl/RAFT)
- [*GMA*](https://github.com/zacjiang/GMA.git)
- [*FlowFormer*](https://github.com/drinkingcoder/FlowFormer-Official)
- [*PWC-Net*](https://github.com/NVlabs/PWC-Net.git)
- [*SpyNet*](https://github.com/sniklaus/pytorch-spynet.git)
- [*FlowNetCRobust*](https://github.com/lmb-freiburg/understanding_flow_robustness)
- [*FlowNet2*](https://github.com/NVIDIA/flownet2-pytorch)
- [*IRR-PWC*](https://github.com/visinf/irr)
- [*SKFlow*](https://github.com/littlespray/SKFlow)
- [*MemFlow*](https://github.com/DQiaole/MemFlow)
- [*RPKNet* (ptlflow)](https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/rpknet)
- [*FlowFormer*](https://github.com/drinkingcoder/FlowFormer-Official)
- [*MatchFlow*](https://github.com/DQiaole/MatchFlow)
- [*GMFlow+* or *unimatch*](https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md\#optical-flow)
- [*SEA-RAFT*](https://github.com/princeton-vl/SEA-RAFT)
- [*MS-RAFT+*](https://github.com/cv-stuttgart/MS_RAFT_plus)
- [*CCMR+*](https://github.com/cv-stuttgart/CCMR)

We thank the original authors for their amazing contributions.

### Additional code

This code base is derived from the repository [PCFA](https://github.com/cv-stuttgart/pcfa).
