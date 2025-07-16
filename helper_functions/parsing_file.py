import argparse


def create_parser(multi_model_csv=False, two_datasets=False, baseline_file=False):
    parser = argparse.ArgumentParser(usage='%(prog)s [options (see below)]')

    # network arguments
    net_args = parser.add_argument_group(title='network arguments')
    if multi_model_csv:
        net_args.add_argument('--ckpt_csv', default='checkpoints_public.csv',
                              help='specify a CSV file containing all model checkpoints to be evaluated')
    else:
        net_args.add_argument('--net', default='RAFT',
                            choices=['RAFT', 'GMA', 'FlowFormer', 'IRR-PWC', 'PWCNet', 'SpyNet', 'FlowNet2', 'FlowNet2C', 'FlowNetC', 'FlowNet2S', 'MemFlow', 'SKFlow', 'unimatch', 'unimatch-scale1', 'unimatch-scale2', 'unimatch-scale2-regrefine6', 'SEA-RAFT-S', 'SEA-RAFT-M', 'SEA-RAFT-L', 'MS-RAFT+', 'CCMR+', 'MatchFlowR', 'MatchFlowG', 'ptlflow-rpknet'],
                            help="specify the network under attack")
        net_args.add_argument('--custom_weight_path',
                            help="specify path to load weights from. By default loads to the given net default weights from `models/_pretrained_weights`")

    # Dataset arguments
    if two_datasets:
        # in-distribution dataset arguments
        dataset_args = parser.add_argument_group(title="in-distribution dataset arguments")
        dataset_args.add_argument('--id_dataset', default='Kitti15', choices=['Kitti15', 'Sintel', 'Spring', 'HD1KSplitScheurer', 'Driving', 'FlyingChairs', 'FlyingThings3D', 'Viper'],
                                help="specify the dataset which should be used for ID evaluation")
        dataset_args.add_argument('--id_dataset_stage', default='training', choices=['training', 'validation'],
                                help="specify the dataset stage ('training' or 'validation') that should be used.")
        # required define for Sintel, FlyingThings3D and Driving 
        dataset_args.add_argument('--id_dataset_pass', default='', choices=['clean', 'final', ''],
                                help="[only sintel,things,driving] specify the rendering pass of the dataset")

        # out-of-distribution dataset arguments
        dataset_args = parser.add_argument_group(title="out-of-distribution dataset arguments")
        dataset_args.add_argument('--ood_dataset', default='Kitti15', choices=['Kitti15', 'Sintel', 'Spring', 'HD1KSplitScheurer', 'Driving', 'FlyingChairs', 'FlyingThings3D', 'Viper'],
                                help="specify the dataset which should be used for OOD evaluation")
        dataset_args.add_argument('--ood_dataset_stage', default='training', choices=['training', 'validation'],
                                help="specify the dataset stage ('training' or 'validation') that should be used.")
        # required define for Sintel, FlyingThings3D and Driving 
        dataset_args.add_argument('--ood_dataset_pass', default='', choices=['clean', 'final', ''],
                                help="[only sintel,things,driving] specify the rendering pass of the dataset")
    else:
        dataset_args = parser.add_argument_group(title="dataset arguments")
        dataset_args.add_argument('--dataset', default='Kitti15', choices=['Kitti15', 'Sintel', 'Spring', 'HD1KSplitScheurer', 'Driving', 'FlyingChairs', 'FlyingThings3D', 'Viper'],
                                help="specify the dataset which should be used for evaluation")
        dataset_args.add_argument('--dataset_stage', default='training', choices=['training', 'validation'],
                                help="specify the dataset stage ('training' or 'validation') that should be used.")
        dataset_args.add_argument('--small_run', action='store_true',
                                help="for testing purposes: if specified the dataloader will on load 32 images")
        
        # required define for Sintel, FlyingThings3D and Driving 
        dataset_args.add_argument('--dataset_pass', default='', choices=['clean', 'final', ''],
                                help="[only sintel,things,driving] specify the dataset type for the sintel dataset")

    # Data saving
    data_save_args = parser.add_argument_group(title="data saving arguments")
    data_save_args.add_argument('--output_folder', default='experiment_data',
                                help="data that is logged during training and evaluation will be saved there")
    if baseline_file:
        data_save_args.add_argument('--baseline_file', default='',
                                    help="file path where to store baseline parameters, defaults to '{output_folder}/baseline_{dataset_id}_{dataset_ood}.json'")
        
    return parser
