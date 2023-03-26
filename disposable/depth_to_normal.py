# %%
from scipy.ndimage import median_filter
from skimage.transform import rescale, resize
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/3detr/third_party/milutils')
import milutils
# %%

def surface_normal(points, sH, sW):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 2
#     lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}

    lookups = {0:(0,-d),1:(d,-d),2:(d,0),3:(d,d),4:(0,d),5:(-d,d),6:(-d,0),7:(-d,-d)}

    surface_normals = np.zeros((sH,sW,3))
    for i in range(sH):
        for j in range(sW):
            min_diff = None
            point1 = points[i,j,:3]
             # We choose the normal calculated from the two points that are
             # closest to the anchor points.  This helps to prevent using large
             # depth disparities at surface borders in the normal calculation.
            for k in range(8):
                try:
                    point2 = points[i+lookups[k][0],j+lookups[k][1],:3]
                    point3 = points[i+lookups[(k+2)%8][0],j+lookups[(k+2)%8][1],:3]
                    diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                    if min_diff is None or diff < min_diff:
                        normal = normalize(np.cross(point2-point1,point3-point1))
                        min_diff = diff
                except IndexError:
                    continue
            surface_normals[i,j,:3] = normal
    return surface_normals

def normalize(v):
    return v/np.linalg.norm(v)

def depth2world(depth_map, intrinsic_param, extrinsic_param, return_full=False):

    # Get world coords
    H, W = depth_map.shape

    WS = np.repeat(np.linspace(1 / (2 * W), 1 - 1 / (2 * W), W).reshape([1, -1]), H, axis=0)
    HS = np.repeat(np.linspace(1 / (2 * H), 1 - 1 / (2 * H), H).reshape([-1, 1]), W, axis=1)

    pixel_coords = np.stack([WS*W, HS*H, np.ones(depth_map.shape)], 2)
    pixel_coords = pixel_coords.reshape(-1, 3).T
    depth_map = depth_map.reshape(-1,1).T
    
    cam_coords = np.linalg.inv(intrinsic_param)@(pixel_coords)
    cam_coords *= depth_map
    
    cam_coords = np.vstack([cam_coords, np.ones((1,cam_coords.shape[1]))])
    world_coords = np.linalg.inv(extrinsic_param)@cam_coords
    
    world_coords = world_coords.T

    # if return_full==False:
    #     mask = np.repeat(depth_map.copy(), 4, axis=0).T
    #     world_coords = world_coords[mask>0].reshape(-1,4)
    #     world_coords = alignment(cls, seq, center, world_coords)
    # else:
    #     world_coords = alignment(cls, seq, center, world_coords)

    return world_coords

def depth_to_surface_normal_opencv_projection(depth, intrinsics, extrinsics, scale=0.25):
    depth_map = depth.copy()
    H, W = depth.shape
    sH, sW = int(scale*H), int(scale*W)
    depth_map[depth < 0.0001] = 50.0

    # Each 'pixel' containing the 3D point in camera coords
    depth_in_world = depth2world(depth_map, intrinsics, extrinsics, True)[:,:3].reshape(H,W,3)
    surface_normals = surface_normal(depth_in_world[::int(1/scale),::int(1/scale),:], sH, sW)
    surface_normals = resize(surface_normals, (H, W), anti_aliasing=True)
    return surface_normals

def generate_SDF(depth, intrinsic_param, extrinsic_param, eta=0.025):

#     depth, mask, intrinsic_param, extrinsic_param, pose = get_default_scene(filename, cls)
    H, W = depth.shape
    # depth[depth>1] = 0 
    depth = median_filter(depth, size=15)

    world_coords = depth2world(depth, intrinsic_param, extrinsic_param)

    surface_normals = depth_to_surface_normal_opencv_projection(depth, intrinsic_param, extrinsic_param)

    WS = np.repeat(np.linspace(1 / (2 * W), 1 - 1 / (2 * W), W).reshape([1, -1]), H, axis=0) * W
    HS = np.repeat(np.linspace(1 / (2 * H), 1 - 1 / (2 * H), H).reshape([-1, 1]), W, axis=1) * H
    s_X = WS[depth > 0]
    s_Y = HS[depth > 0]

    pos_pts_world = world_coords[:,:3].copy()
    neg_pts_world = world_coords[:,:3].copy()
    # purturb with surface normal
    for idx, (xx,yy) in enumerate(zip(s_X,s_Y)):
        pos_pts_world[idx] += eta * np.array(surface_normals[int(yy)][int(xx)])
        neg_pts_world[idx] -= eta * np.array(surface_normals[int(yy)][int(xx)])

    eta_vec = np.ones((pos_pts_world.shape[0], 1)) * eta
    part = np.zeros((pos_pts_world.shape[0], 1))
    pos = np.hstack([pos_pts_world, eta_vec, part])
    neg = np.hstack([neg_pts_world, -eta_vec, part])
    
    return pos, neg, world_coords[:,:3]


# %%
h5_depth = h5py.File('/home/mil/kawana/workspace/3detr/third_party/OPD/dataset/SapienDataset_h5/depth.h5', 'r')
# %%
depth = h5_depth['depth_images'][0, ..., 0]
# plt.imshow(depth)
# plt.show()
intrinsic_param = np.array([305.2046203613281,
    0.0,
    322.61651611328125,
    0.0,
    305.2046203613281,
    179.24960327148438,
    0.0,
    0.0,
    1.0]).reshape([3,3])
pos, neg, coords = generate_SDF(depth, intrinsic_param, np.eye(4))

perm = np.random.permutation(pos.shape[0])
pos = pos[perm][:5000]
fig = milutils.visualizer.get_pcd_plot(pos[:, :3])
fig.show()