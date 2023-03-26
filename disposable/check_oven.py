# %%
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import trimesh

split = 'test'
if split == 'train':
    split_ = 'train'
else:
    split_ = 'val'

train_path = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/derivatives/3detr/splits/oc0.5_tr0.25_str0.95-0-objs4-train20000test4000/val.json'
with open(train_path) as f:
    paths = json.load(f)

for path in tqdm(paths):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    if not 'oven' in pkl['data_tracing']['categories']:
        continue
    objs = defaultdict(lambda: 0)
    for bidx, box in enumerate(pkl['boxes']):
        if box['category'] != 'oven':
            continue
        objs[box['obj_id_in_scene']] += 1
    for k, v in objs.items():
        assert v != 2



