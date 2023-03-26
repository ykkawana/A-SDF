# %%
import json
import pickle
from collections import defaultdict
from tqdm import tqdm

cats = defaultdict(lambda: [])
for split in ['val']:
    train_path = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/derivatives/3detr/splits/oc0.5_tr0.25_str0.95-0-objs4-train20000test4000/{split}.json'

    with open(train_path) as f:
        paths = json.load(f)

    for pidx, path in tqdm(enumerate(paths)):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        objs = defaultdict(lambda: 0)
        for bidx, box in enumerate(data['boxes']):
            objs[(box['category'], box['obj_id'], box['obj_id_in_scene'])] += 1
        for k, v in objs.items():
            cats[k[0]].append(v) 
    

# %%
import numpy as np

for k, v in cats.items():
    freq = defaultdict(lambda: 0)
    for vv in v:
        freq[vv] += 1
    keys = np.array(list(freq.keys()))
    vals = np.array(list(freq.values()))
    ma = np.argmax(vals)
    makey = keys[ma]
    print(k, makey, freq[makey], len(v))
    ma2 = np.argsort(vals)[::-1][:2]
    print(k, vals[ma2], keys[ma2])
    # print(k, keys, vals)
    """
    dishwasher 2 4867 4997
    trashcan 2 4721 5037
    safe 2 4795 4795
    oven 2 2803 4951
    storagefurniture 2 1748 4907
    table 2 2005 4790
    microwave 2 3847 4832
    refrigerator 3 3326 5089
    washingmachine 2 5042 5042
    box 2 2902 4823   
    """