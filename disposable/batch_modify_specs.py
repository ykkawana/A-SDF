# %%
import glob
import os
import json
import shutil

paths = glob.glob('/home/mil/kawana/workspace/3detr/third_party/A-SDF/examples/sapien_fixed/*/specs.json')

# %%
for path in paths:
    with open(path) as f:
        specs = json.load(f)
    specs['TrainSplit'] = './sapien_shapes_train_fixed_max1500.pkl'
    specs['TestSplit'] = './sapien_shapes_test_fixed.pkl'

    old_path = path.replace('.json', '.bk.json')
    shutil.copy2(path, old_path)
    with open(path, 'w') as f:
        json.dump(specs, f, indent=4)
# %%
