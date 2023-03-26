# %%
import numpy as np

path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/data/NormalizationParameters/shape2motion/refrigerator/0006art00400040.npz'
npz = np.load(path)
# %%
for k, v in npz.items():
    print(k, v)

# %%
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/3detr/third_party/milutils')
import milutils

path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/data/SdfSamples/shape2motion/refrigerator/0006art00900090.npz'
npz = np.load(path)
for k, v in npz.items():
    print(k, v.shape)
# xyz [-1, 1]
# pos is outside
# neg is inside
key = 'neg'
xyz = npz[key][:, :3]
print(xyz.min(0), xyz.max(0))
print(np.linalg.norm(xyz.max(0) - xyz.min(0)))
# %%
dist = npz[key][:, 3]
label = npz[key][:, 4]
print(label.min())
print(label.max())
# %%
perm = np.random.permutation(xyz.shape[0])[:5000]
xyz = xyz[perm]
dist = dist[perm]
label = label[perm]

ls = np.unique(label)
ps = []
for l in ls:
    ps.append(xyz[label == l])

fig = milutils.visualizer.get_pcd_plot(ps)
fig.show()
# %%
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.chdir('/home/mil/kawana/workspace/3detr/third_party/A-SDF')
import sys
sys.path.insert(0, '.')
from asdf.data import SapienSDFSamples
import asdf.workspace as ws

import sys
sys.path.insert(0, '/home/mil/kawana/workspace/3detr/third_party/milutils')
import milutils
import numpy as np
# %%
## Visualize dataset
experiment_directory = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/examples/sapien_fixed/table'
experiment_directory = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/examples/sapien_fixed/microwave-2'
specs = ws.load_experiment_specifications(experiment_directory)
data_source = specs["DataSource"]
train_split_file = specs["TrainSplit"]
test_split_file = specs["TestSplit"]

num_samp_per_scene = specs["SamplesPerScene"]

sdf_dataset = SapienSDFSamples(
    data_source, train_split_file, num_samp_per_scene, specs['Class'], load_ram=False, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], art_per_instance=specs["ArticulationPerInstance"])

sdf_dataset_test = SapienSDFSamples(
    data_source, test_split_file, num_samp_per_scene, specs['Class'], load_ram=False, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], art_per_instance=specs["ArticulationPerInstance"], fixed_articulation_type='fixed')
# %%
data = sdf_dataset[3]
# data2 = sdf_dataset_test[5]

d = data
perm = np.random.permutation(d[0][0].shape[0])#[:5000]
xyz = d[0][0][:, :3].numpy()
dist = d[0][0][:, 3].numpy()
label = d[0][0][:, 4].numpy()
xyz = xyz[perm]
dist = dist[perm]
label = label[perm]

ls = np.unique(label)
ps = []
for l in ls:
    if l == 0:
        continue
    ps.append(xyz[label == l])

fig = milutils.visualizer.get_pcd_plot(ps)
fig.show()
# %%
