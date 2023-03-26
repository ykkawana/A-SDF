# %%
import pickle
from collections import defaultdict
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
for split in ['train']:
    path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f)['shapes']
    for cat, cat_vals in data.items():
        print(cat, len(cat_vals))
        for obj_key, obj in cat_vals.items():
            assert len(obj['bboxes']) == cat_num[cat] + 1, (cat,obj_key,len(obj['boxes']))

            pcnts = defaultdict(lambda: 0)
            bcnt = 0
            for oo in obj['bboxes']:
                pcnts[oo['part_cnt']] += 1
                if oo['type'] == 2:
                    bcnt += 1
            for v in pcnts.values():
                assert v == 1
            assert bcnt == 1

# %%
