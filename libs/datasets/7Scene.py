from glob import glob
import os

from .dataset import Dataset
from libs.general.utils import import *

class Scenes7(Dataset):

    def __init__(self, *args, **kwargs):
        super(Scenes7).__init__(*args, **kwargs)
        self.w, self.h = 640, 480
        self.ox, self.oy = 320, 240
        self.f = 585
        self.K = np.array([[self.f, 0, self.ox], [0, self.f, self.oy], [0, 0, 1]], dtype=np.float32)

    def synchronize_timestamps(self):
        """Synchronize RGB, Depth, and Pose timestamps to form pairs

        Returns:
            a dictionary containing
                - **rgb_timestamp** : {'depth': depth_timestamp, 'pose': pose_timestamp}
        """
        self.rgb_d_pose_pair = {}
        len_seq = len(glob(os.path.join(self.data_dir['img'], "*.{}".format(self.cfg.image.ext))))
        for i in range(len_seq):
            self.rgb_d_pose_pair[i] = {}
            self.rgb_d_pose_pair[i]['depth'] = i
            self.rgb_d_pose_pair[i]['pose'] = i

    def get_timestamp(self, img_id):
        """Get timestamp for the query img_id

        Args:
            img_id (int): query image id

        Returns:
            timestamp (int): timestamp for query image
        """
        return img_id

    # def get_intrinsics_param(self):

    def get_data_dir(self):

        data_dir = {}

        img_seq_dir = os.path.join(
                            self.cfg.directory.img_seq_dir,
                            self.cfg.seq
                            )
        data_dir['img'] = os.path.join(img_seq_dir)

        # get depth data directory
        data_dir['depth_src'] = self.cfg.depth.depth_src

        if data_dir['depth_src'] == 'gt':
            data_dir['depth'] = os.path.join(self.cfg.directory.depth_dir)
        elif data_dir['depth_src'] is None:
            data_dir['depth'] = None
        else:
            assert False, "Wrong depth src [{}] is given.".format(data_dir['depth_src'])

        return data_dir

    def get_gt_poses(self):
        """Get ground-truth poses

        Returns:
            gt_poses (dict): each pose is a [4x4] array
        """
        annotations = os.path.join(
                            self.cfg.directory.gt_pose_dir,
                            "{}.txt".format()
                            )
        gt_poses = load_poses_from_txt(annotations)
        return gt_poses

    def get_depth(self, timestamp):
        """Get GT/precomputed depth data given the timestamp

        Args:
            timestamp (int): timestamp for the depth

        Returns:
            depth (array, [HxW]): depth data
        """
        img_id = self.rgb_d_pose_pair[timestamp]['depth']

        img_name = "frame-{:6d}.png".format(img_id)
        sacle_factor = 500

        img_h, img_w = self.cfg.image.height, self.cfg.image.width
        depth_path = os.path.join(self.data_dir['depth'], img_name)
        depth = read_depth(depth_path, scale_factor, [img_h, img_w])
        return depth



