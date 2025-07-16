import json
import os
import numpy as np
import pandas as pd

root = 'paper_results/accuracy'

results = []


def store_metrics(json_filepath, meta_data):
    global results

    with open(json_filepath, 'r') as f:
        json_obj = json.load(f)
    for m in [
        "aee_mean",
        "fl-epe-1px_mean",
        "fl-epe-3px_mean",
        "fl-epe-5px_mean",
        "fl-all-kitti_mean",
        "mse_mean",
        "wauc_mean",
        "runtime_s",
    ]:
        if m in json_obj:
            new_entry = {
                **meta_data,
                'metric_name': m,
                'metric_value': json_obj[m],
                'weight_path': json_obj['config']['weight_path']
            }

            if new_entry not in results:
                results.append(new_entry)
                
            assert not pd.isna(m), (m)
            assert not pd.isna(json_obj[m]), (m, json_obj)


for dataset_name in os.listdir(root):
    dataset_path = os.path.join(root, dataset_name)
    for datastage_or_pass_name in os.listdir(dataset_path):
        datastage_path = os.path.join(dataset_path, datastage_or_pass_name)
        
        if datastage_or_pass_name in ['clean', 'final']:
            stages = os.listdir(datastage_path)
            assert len(stages) == 1, (stages, datastage_path)
            datastage_path = os.path.join(datastage_path, stages[0])
            pass_name = datastage_or_pass_name
            datastage_or_pass_name = stages[0]
        else:
            print('assuming final pass', dataset_name)
            pass_name = ''
            

        for net_name in os.listdir(datastage_path):
            net_path = os.path.join(datastage_path, net_name)

            if net_name == 'SpyNet':
                store_metrics(
                    os.path.join(net_path, 'metrics.json'),
                    meta_data={
                        'dataset': dataset_name,
                        'dataset_stage': datastage_or_pass_name,
                        'dataset_pass': pass_name,
                        'model': net_name,
                        'checkpoint': 'default',
                    }
                )
                continue

            for weights_name in os.listdir(net_path):
                if weights_name == 'timestamped':
                    continue
                weights_path = os.path.join(net_path, weights_name)

                metrics_path = os.path.join(weights_path, 'metrics.json')
                if '/old/' in metrics_path:
                    continue

                if not os.path.exists(metrics_path):
                    print('[Warning] ', metrics_path, ' does not exist')
                    continue

                store_metrics(
                    metrics_path,
                    meta_data={
                        'dataset': dataset_name,
                        'dataset_stage': datastage_or_pass_name,
                        'dataset_pass': pass_name,
                        'model': net_name,
                        'checkpoint': weights_name,
                    }
                )

df = pd.DataFrame(results)
df1 = df[df[[c for c in df.columns if c not in ['metric_value', 'dataset_pass']]].isna().any(axis=1)]
print(df1)
for x in results:
    cols = [c for c in df.columns if c != 'metric_value']
x = df.value_counts([c for c in df.columns if c != 'metric_value'], dropna=False)

print('duplicates \n', x[x > 1])

_df_without_dup = df.drop_duplicates([c for c in df.columns if c != 'metric_value'])
assert df.shape == _df_without_dup.shape, (df.shape, _df_without_dup.shape)

os.makedirs('experiment_data', exist_ok=True)
fname = 'experiment_data/summary_metrics.csv'
df.to_csv(fname, index=False)
print('written to ', fname)
