import os.path as osp
from pathlib import Path
from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from read import get_depth_path
from torchvision.transforms import (
    Compose,
    ToTensor,
    Resize,
    GaussianBlur,
    CenterCrop,
    ColorJitter,
    RandomCrop,
)
from libs.datasets.read import read_poses, get_depth, read_depth
from libs.geometry.camera_modules import Intrinsics

def get_indexs(image_num):
    def _get_group_from_range(start, end):
        group = []
        for i in range(start, end-1):
            group.append([i, i+1])
        return group

    groups = _get_group_from_range(0, image_num)
    # print("Groups of consecutive image pairs for training:", groups)
    return groups


class Index(Dataset):
    def __init__(self, indexes):
        super(Index, self).__init__()

        self.indexes = indexes

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        index = int(self.indexes[index])
        return index


# ============================
def get_training_data(data_dir):
    scene_path = osp.expanduser(data_dir)
    train_path = osp.join(scene_path, "dataset_train.txt")

    seq_ids, img_ids, tvecs, rotmats, file_paths = read_poses(train_path, "7Scenes")

    # Concatenate Pose
    poses = torch.from_numpy(
        np.concatenate((tvecs, rotmats.reshape(-1, 9)), axis=1)
    ).to(dtype=torch.float32)

    groups = get_indexs((len(img_ids)))
    indexes = [str(i) for i in range(len(groups))]

    dataset = Index(indexes)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1, shuffle=False, pin_memory=True)

    # transform = create_transforms(640, 480, True, 224)

    # all_images = []
    all_poses = []
    # all_depths = []
    all_depth_paths = []
    all_image_ids = []
    all_image_paths = []

    print("begin getting index")
    for _, index in enumerate(dataloader):
        print("curr", index)
        group = groups[index]

        group_image_path = [file_paths[i] for i in group]
        group_pose = [poses[i] for i in group]
        group_seq_id = [seq_ids[i] for i in group]
        group_img_id = [img_ids[i] for i in group]

        batch_images = []
        batch_poses = []
        batch_depths = []
        batch_depth_paths = []
        batch_image_ids = []
        batch_image_paths = []

        for i in range(2):
            image_path = group_image_path[i]
            image_path = osp.join(scene_path, image_path)
            print(type(image_path))
            # image = transform(Image.open(image_path))
            # depth = get_depth(scene_path, group_seq_id[i], group_img_id[i])
            # depth = torch.from_numpy(depth).to(dtype=torch.float32)
            depth_path = get_depth_path(scene_path, group_seq_id[i], group_img_id[i])

            batch_image_ids.append(group_img_id[i])
            # batch_images.append(image)
            # batch_depths.append(depth)
            batch_depth_paths.append(depth_path)
            batch_poses.append(group_pose[i])
            batch_image_paths.append(image_path)

        # all_images.append(batch_images)
        all_poses.append(batch_poses)
        # all_depths.append(batch_depths)
        all_image_ids.append(batch_image_ids)
        all_image_paths.append(batch_image_paths)
        all_depth_paths.append(batch_depth_paths)

    return all_image_paths, all_poses, all_depth_paths, all_image_ids


class SevenScenes(Dataset):
    """Dataset class for the 7-Scenes dataset."""

    def __init__(
        self,
        split_file_path: Path,
        image_size: int,
        mode: str = "resize",
        half_image: bool = False,
        crop: Optional[float] = None,
        gauss_kernel: Optional[int] = None,
        gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
        jitter_brightness: Optional[float] = None,
        jitter_contrast: Optional[float] = None,
        jitter_saturation: Optional[float] = None,
        jitter_hue: Optional[float] = None,
    ) -> None:

        print("Preparing dataset 7Scenes...")

        K_params = [320, 240, 585, 585]
        self.cam_intrinsics = Intrinsics(K_params)


        self._images_paths, self._poses, self._depth_paths, self._img_ids = get_training_data(split_file_path)

        self._transform = create_transforms(
            *ToTensor()(Image.open(self._images_paths[0][0])).shape[1:],
            True,
            image_size,
            mode,
            crop,
            gauss_kernel,
            gauss_sigma,
            jitter_brightness,
            jitter_contrast,
            jitter_saturation,
            jitter_hue,
        )

    def __len__(self) -> int:
        """Get total number of samples."""
        return len(self._img_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get sample by index.

        Args:
            sample index

        Returns:
            image tensor, shape (3, H, W),
            image pose, shape (12,)
                formatted as (tx, ty, tz, r11, r12, r13, r21, r22, r23, r31, r32, r33),
            image name
        """
        return (
            # self._images_paths[index],
            self._transform(Image.open(self._images_paths[index][0])),
            self._transform(Image.open(self._images_paths[index][1])),
            read_depth(self._depth_paths[index][0], 500),
            read_depth(self._depth_paths[index][1], 500),
            self._poses[index],
            self._img_ids[index],
        )




def create_transforms(
    height: int,
    width: int,
    to_tensor: bool,
    image_size: int,
    mode: str = "resize",
    crop: Optional[float] = None,
    gauss_kernel: Optional[int] = None,
    gauss_sigma: Optional[Union[float, Tuple[float, float]]] = None,
    jitter_brightness: Optional[float] = None,
    jitter_contrast: Optional[float] = None,
    jitter_saturation: Optional[float] = None,
    jitter_hue: Optional[float] = None,
) -> Compose:
    """Create transforms.

    Args:
        height: original height of images
        width: original width of images
        to_tensor: if True, ToTensor transform is included
        image_size: image size (images are made square)
        mode: how image is made smaller,
            options: "resize", "random_crop", "vertical_crop", "center_crop", "posenet"
            mode "posenet" overrides image_size and crop to that of posenet
        crop: what ratio of original height and width is cropped
        gauss_kernel: size of Gaussian blur kernel
        gauss_sigma: [min and max of] std dev for creating Gaussian blur kernel
        jitter_brightness: brightness jitter
        jitter_contrast: contrast jitter
        jitter_saturation: saturation jitter
        jitter_hue: hue jitter
    """
    transforms = [ToTensor()] if to_tensor else []
    if mode == "posenet":
        print(
            "Following PoseNet images are resized to smallest edge 256,"
            + " then random cropped to 224x224"
        )
        transforms.extend([Resize(256, antialias=True), RandomCrop(224)])

    else:
        if crop is not None:
            print(f"Images are cropped by {crop} of its height and width")
            transforms.append(RandomCrop((int(crop * height), int(crop * width))))
        if mode == "resize":
            print(f"Images are resized to {image_size}x{image_size}")
            transforms.append(Resize((image_size, image_size), antialias=True))
        elif mode == "center_crop":
            print(f"Images are cropped such that smallest edge is {image_size}")
            transforms.extend([CenterCrop(256), RandomCrop(224)])
        elif mode == "random_crop" or mode == "vertical_crop":
            if mode == "vertical_crop":
                print(f"Images are cropped such that smallest edge is {image_size}")
                transforms.append(Resize(image_size, antialias=True))
            print(f"Images are random cropped to {image_size}x{image_size}")
            transforms.append(RandomCrop(image_size))
        elif mode == "ceiling_train":
            print(f"Images are cropped for Ceiling training.")
            transforms.extend([CenterCrop(256), RandomCrop(224)])
        elif mode == "ceiling_test":
            print(f"Images are cropped for Ceiling test.")
            transforms.extend([CenterCrop(224)])
        else:
            raise ValueError("Invalid image resizing mode.")

    if gauss_kernel is not None and gauss_sigma is not None:
        print(
            f"Gaussian blur of kernel size {gauss_kernel}"
            + f" and std dev {gauss_sigma} is applied"
        )
        transforms.append(GaussianBlur(gauss_kernel, sigma=gauss_sigma))
    if (
        jitter_brightness is not None
        and jitter_contrast is not None
        and jitter_saturation is not None
        and jitter_hue is not None
    ):
        print(
            f"Color jitter of {jitter_brightness} brightness,"
            + f" {jitter_contrast} contrast,"
            + f" {jitter_saturation} saturation, and {jitter_hue} hue is applied"
        )
        transforms.append(
            ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            )
        )
    return Compose(transforms)

