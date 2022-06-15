# -- ARCHIVED --
# upload / download models and logs to tensorboard

# imports
import os
import shutil

import pandas as pd
import tensorboard as tb

# copy relevant model metrics into seperate folder and upload to tensorboard dev
data_filter = 'test' # 'train', 'test' or 'validation'
model_filter = 'resnet'# 'basic' or 'resnet'
log_path = '../logs/scalars'
upload_path = '../tensorboard_upload/'
os.mkdir(upload_path)

# iterate through all files in log_path and copy relevant to upload_path
for (dirpath, dirnames, filenames) in os.walk(log_path):
    # filter dirs
    if data_filter in dirpath and model_filter in dirpath:

        from_directory = dirpath
        to_directory = upload_path + dirpath[16:]

        # copy dir trees
        shutil.copytree(from_directory, to_directory)

# TODO: run in project root for upload
print(f'tensorboard dev upload --logdir=./tensorboard_upload/ --name FLE_{model_filter}_{data_filter}')
# FIXME: wait for user imput to continue

# remove copied dir trees
shutil.rmtree(upload_path)

# TODO: set tensorboard dev ID
experiment_ids = ['0m4NvD5jTxmSw3hr8SeJww', 'uLYWP18zQ1eMtDBiLHLHrg'] # [resnet, basic]
all_dfs = []

# download experiments
for ind, id in enumerate(experiment_ids):
    experiment = tb.data.experimental.ExperimentFromDev(id)
    df_temp = experiment.get_scalars()
    
    print(f'{ind}: {df_temp.shape}')
    all_dfs.append(df_temp)

# combine experiments into single dataframe
df = all_dfs[0]

if len(all_dfs) > 1:
    for new_df in all_dfs[1:]:
        df = pd.concat([df, new_df])

print(df.shape)
df.head()
