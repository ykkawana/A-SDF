# %%
import pickle
from collections import defaultdict
import random
random.seed(0)

additional_only = True
cat_num = dict(
    dishwasher=1,
    trashcan=1,
    safe=1,
    oven=2,
    storagefurniture=1,
    table=1,
    microwave=1,
    refrigerator=2,
    washingmachine=1,
    box=1,
)
cat_num_additional = dict(
    storagefurniture=2,
    trashcan=2,
    microwave=2,
    box=4,
    refrigerator=1,
    table=2,
)
ma = 1500
for split in ['train']:
    if additional_only:
        path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed_additional.pkl'
    else:
        path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed.pkl'

    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    data = pkl['shapes']
    new_data = {}
    for cat, cat_vals in data.items():
        if additional_only:
            if cat not in cat_num_additional:
                continue
            cat_vals = {k: v for k, v in cat_vals.items() if len(v['bboxes']) == cat_num_additional[cat] + 1}
        print(cat, len(cat_vals))
        if len(cat_vals) > ma:
            new_keys = random.sample(list(cat_vals.keys()), ma)
            new_d = {k: cat_vals[k] for k in new_keys}
            cat_vals = new_d
        new_data[cat] = cat_vals
    data = new_data
    print()
    print('sampled')
    print()
    for cat, cat_vals in data.items():
        print(cat, len(cat_vals))
    pkl['shapes'] = data
    if additional_only:
        path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed_max1500_additional_only.pkl'
    else:
        path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed_max1500.pkl'
    with open(path, 'wb') as f:
        pickle.dump(pkl, f)
# %%
