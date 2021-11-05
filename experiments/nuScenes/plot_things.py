import sys
sys.path.append('../../trajectron')
import os
import numpy as np
import torch
import dill
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe
from helper import *
import visualization

nuScenes_data_path = '/home/zhud/dataset/v1.0-trainval' # Data Path to nuScenes data set
# nuScenes_devkit_path = './devkit/python-sdk/'
# sys.path.append(nuScenes_devkit_path)
from nuscenes.map_expansion.map_api import NuScenesMap
nusc_map = NuScenesMap(dataroot=nuScenes_data_path, map_name='boston-seaport')

line_colors = ['#375397','#80CBE5','#ABCB51','#F05F78', '#C8B0B0']


with open('/home/zhud/dataset/test_1/nuScenes_test_full.pkl', 'rb') as f:
    eval_env = dill.load(f, encoding='latin1')
eval_scenes = eval_env.scenes


log_dir = 'models/'
model_dir = os.path.join(log_dir, 'int_ee_me')
eval_stg_traj, _ = load_model(model_dir, eval_env, ts=12)

log_dir = '/home/zhud/scratch/motionsafe/log/'
model_dir = os.path.join(log_dir, 'log0.000000001_25_Aug_2020_03_08_32_510')
eval_stg_our, _ = load_model(model_dir, eval_env, ts=16)

# eval_stg_before, _ = load_model(model_dir, eval_env, ts=15)
# eval_stg_after, _ = load_model(model_dir, eval_env, ts=8)


ph = 6
save_path = '../../logs/img/log'
# plot_one(eval_stg_our, eval_scenes[13], ph, 'normal')
# for i in range(15):
for i, scene in enumerate(eval_scenes[0:20]):
    # plot_one(eval_stg_before, scene, ph, 'before', save_path)
    # plot_one(eval_stg_after, scene, ph, 'after', save_path)

    plot_kde(eval_stg_traj, scene, ph, 'traj', save_path)['VEHICLE']
    plot_kde(eval_stg_our, scene, ph, 'our', save_path)['VEHICLE']

