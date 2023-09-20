import torch
from SevenScene import SevenScenes, get_training_data
from torch.utils.data import Dataset
import os.path as osp
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--half_image", action="store_true")
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument("--image_mode", type=str, default="ceiling_train")
parser.add_argument("--image_crop", type=float, default=None)
parser.add_argument("--gauss_kernel", type=int, default=3)
parser.add_argument("--gauss_sigma", type=float, nargs="*", default=(0.05, 5.0))
parser.add_argument("--jitter_brightness", type=float, default=0.7)
parser.add_argument("--jitter_contrast", type=float, default=0.7)
parser.add_argument("--jitter_saturation", type=float, default=0.7)
parser.add_argument("--jitter_hue", type=float, default=0.5)
args = parser.parse_args()

kwargs = {
    "image_size": args.image_size,
    "mode": args.image_mode,
    "half_image": args.half_image,
    "crop": args.image_crop,
    "gauss_kernel": args.gauss_kernel,
    "gauss_sigma": args.gauss_sigma,
    "jitter_brightness": args.jitter_brightness,
    "jitter_contrast": args.jitter_contrast,
    "jitter_saturation": args.jitter_saturation,
    "jitter_hue": args.jitter_hue,
}

scene_path = osp.expanduser("/Users/seungjuuu/Downloads/chess/")
# images, poses, depths, id = get_training_data(scene_path)

train_set = SevenScenes(scene_path, **kwargs)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=3, shuffle=True)
cur_data = {}
ref_data = {}

print('haha',train_set.cam_intrinsics.mat)

for _, batch in enumerate(train_loader):
    print('4', batch[4])
    print('id',batch[5])
    print('pose',batch[4][0])
    print('shape',batch[4][0].shape)
    pose1 = batch[4][0]
    print('3',pose1[:,:3])
    # pose1[:,:3] = torch.tensor([0,1,2])
    # print('new',pose1)





#
# train_set = SevenScenes(scene_path)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
# for _, batch in enumerate(train_loader):
#     for item in batch:
#         item_type = type(item)
#         if not isinstance(item, (torch.Tensor, np.ndarray, int, float, dict, list)):
#             print(f"Unsupported data type found: {item_type}")
