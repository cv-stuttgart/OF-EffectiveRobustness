
import pandas as pd
import numpy as np

REPO_PATH = '../..'

FIX_WORDING = dict([
    ('FlyingThings3DFinal', 'Things'),
    ('HD1KSplitScheurer', 'HD1K'),
    ('DrivingSample', 'Driving'),
    ('Kitti15', 'KITTI'),
    ('Viper', 'VIPER'),
    ('training', 'train'),
    ('evaluation', 'validation'),
    ('unimatch-scale2-regrefine6', 'GMFlow+'),
    ('unimatch-scale2', 'GMFlow+ (s2)'),
    ('unimatch-scale1', 'GMFlow+ (s1)'),
    ('MatchFlowR', 'MatchFlow (R)'),
    ('MatchFlowG', 'MatchFlow (G)'),
    ('ptlflow-rpknet', 'RPKNet'),
    ('SEA-RAFT-S', 'SEA-RAFT (S)'),
    ('SEA-RAFT-M', 'SEA-RAFT (M)'),
    ('SEA-RAFT-L', 'SEA-RAFT (L)'),
    ('mse_mean', r'MSE'),
    ('wauc_mean', 'WAUC'),
    ('aee_mean', 'EPE'),
    ('num_params', r'\#weights'),
    ('fl-epe-1px_mean', r'IR_{1px}'),
    ('fl-epe-3px_mean', r'IR_{3px}'),
    ('fl-epe-5px_mean', r'IR_{5px}'),
    ('fl-all-kitti_mean', r'fl-kitti'),
    ('EPE-robustness', r'ER_{EPE}'),
    ('WAUC-robustness', r'ER_{WAUC}'),
    # ('runtime_s', 'Runtime [s]'),
])


def make_column(df: pd.DataFrame, transpose_columns: list, value_column='metric_value'):
    """(copy pasted from I_thesis code)"""
    df = df.copy()
    df[transpose_columns] = df[transpose_columns].replace(np.nan, '')
    column_names = list(
        tuple(x) for i, x in df[transpose_columns].drop_duplicates().iterrows())
    row_idx = [x for x in df.columns if x not in [
        *transpose_columns, value_column]]

    df_ = df[row_idx].drop_duplicates()
    for ds in column_names:
        # print(df_.head())

        df_next = df.copy()
        for i, x in enumerate(ds):
            df_next = df_next[df_next[transpose_columns[i]] == x]
        df_next = df_next.drop(transpose_columns, axis=1)
        df_next = df_next.rename(columns={value_column: ' '.join(ds)})
        # if len(ds) == 1:
        #     df_next = df_next.rename(columns={value_column: ds[0]})
        # else:
        #     df_next = df_next.rename(columns={value_column: ds})

        df_ = df_.merge(df_next, on=row_idx,
                        how='outer',
                        suffixes=['', '_' + '_'.join(ds)],
                        validate='one_to_one'
                        )

    return df_

# normalize weightpath
def normalize_weightpath(row):
        if pd.isna(row['weight_path']):
            return ''
        weight_path = row['weight_path']
        weight_path = weight_path.replace(
            'models/_pretrained_weights/../../', '')

        weight_path = weight_path.replace(
            '/home/bauerkn/model_repos/unimatch/pretrained/', 'models/_pretrained_weights/unimatch_weights/')

        str_idx = weight_path.find('models/_pretrained_weights')
        if str_idx != -1:
            weight_path = weight_path[str_idx:]
        elif weight_path.startswith('../'):
            weight_path = weight_path.replace('../', '/home/bauerkn/')
        elif weight_path.startswith('~/'):
            weight_path = weight_path.replace('~/', '/home/bauerkn/')
        elif '/model_repos/raft_plain/' in weight_path:
            weight_path = weight_path.replace('/model_repos/raft_plain/', '/raft_plain/')
        elif weight_path in ['chairs', 'things', 'sintel', 'kitti', '']:
            # RPKNet checkpoints
            pass
        else:
            assert weight_path.startswith('/'), weight_path

        return weight_path


def prepare_separate():

    for training_stage in ['things', 'sintel', 'kitti']:

        df_metrics = pd.read_csv(f'{REPO_PATH}/experiment_data/summary_metrics.csv')

        df_metrics['weight_path'] = df_metrics.apply(normalize_weightpath, axis=1)
        df_checkpoints_meta = pd.read_csv(f'{REPO_PATH}/config/checkpoints_public.csv')
        print(df_metrics.head(1))
        print(df_checkpoints_meta.head(1))


        df = df_metrics.merge(df_checkpoints_meta, how='inner', left_on=['model', 'weight_path'], right_on=['net', 'weight_path'])
        # print(df.head(1))

        df = df.drop(columns=['dataset_stage', 'dataset_pass', # redundant due to uniqueness per dataset
                            'net', # redundant due to merge
                            'weight_path', 'checkpoint', 'training_datasets', # too many columns
                            ])
        

        # filtering (would be nice to have it interactively in the website :-) )
        df = df[df['metric_name'] == 'wauc_mean']# .endswith('_mean')]
        df = df[df['training_stage'] == training_stage]
        df['metric_name'] = df['metric_name'].str.replace('_mean', '').str.upper()
        df = df.drop(columns=['training_stage', 'metric_name'])


        # display of WAUC
        df['metric_value'] = (df['metric_value'] * 100).round(2)

        df = make_column(df, transpose_columns=['dataset'], value_column='metric_value')

        df = df[
            ['model',
            'FlyingThings3D',
            'Sintel',
            'Kitti15',
            'HD1KSplitScheurer',
            'Driving',
            'Viper',
            'Spring',
            ]
        ]

        df = df.sort_values(by=['FlyingThings3D'], ascending=False)

        # rename columns
        df = df.replace(FIX_WORDING)
        df = df.rename(columns=FIX_WORDING)
        df = df.rename(columns={
            'model': 'Model',
            # 'Kitti15': 'KITTI 2015',
            # 'HD1KSplitScheurer': 'HD1K',
        })

        print(df.head(10))

        df.to_csv(f'../static/data/summary_metrics_{training_stage}.csv', index=False)

        if training_stage == 'things':
            df.to_csv(f'../static/data/summary_metrics.csv', index=False)


def prepare_all_in_one():
    
    df_metrics = pd.read_csv(f'{REPO_PATH}/experiment_data/summary_metrics.csv')

    df_metrics['weight_path'] = df_metrics.apply(normalize_weightpath, axis=1)
    df_checkpoints_meta = pd.read_csv(f'{REPO_PATH}/config/checkpoints_public.csv')
    print(df_metrics.head(1))
    print(df_checkpoints_meta.head(1))
    
    
    df = df_metrics.merge(df_checkpoints_meta, how='inner', left_on=['model', 'weight_path'], right_on=['net', 'weight_path'])
    
    df = df[['dataset','dataset_stage','dataset_pass','training_stage','checkpoint','metric_name','metric_value','weight_path','net']]
    df = df[df['metric_name'] != 'runtime_s']# Runtime depends on hardware and we used varying hardware
    
    
    fix_wording = FIX_WORDING.copy()
    df = df.replace(fix_wording)
    df = df.rename(columns=fix_wording)
    df['training_stage'] = df['training_stage'].str.replace('things', 'C+T').replace('sintel', 'S').replace('kitti', 'K').replace('chairs', 'C')


    df.to_csv(f'../static/data/summary_metrics_merged.csv', index=False)

# prepare_separate()
prepare_all_in_one()

print('Done')
