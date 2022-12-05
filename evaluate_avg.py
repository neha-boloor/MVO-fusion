
import argparse
import os.path as osp

import numpy as np
import glob

from Datasets.utils import plot_traj
from evaluator.tartanair_evaluator import TartanAirEvaluator


def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--nuscenesmini', action='store_true', default=False,
                        help='nuscenes-mini test (default: False)')
    parser.add_argument('--nuscenes', action='store_true', default=False,
                        help='nuscenes test (default: False)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    # load trajectory data from a folder
    datastr = None
    if args.nuscenesmini:
        datastr = 'nuscenesmini'
    elif args.nuscenes:
        datastr = 'nuscenes'

    scene_dir = glob.glob(osp.join("results", datastr, "*"))
    for scene in scene_dir:
        if not osp.isdir(scene):
            continue

        avg_quaternion = osp.join(scene, "average_transform_estimate", "avg_est.npy")
        eul_mean = osp.join(scene, "average_transform_estimate", "eul_mean_est.npy")
        eul_median = osp.join(scene, "average_transform_estimate", "eul_median_est.npy")
        avg_gt = osp.join(scene, "average_transform_estimate", "avg_gt.txt")

        fusion_types = ("avg_quaternion", "eul_mean", "eul_median")
        for fusion_type in fusion_types:
            pred_file = locals()[fusion_type]
            gt_file = avg_gt

            # calculate ATE, RPE, KITTI-RPE
            poselist = np.load(pred_file)
            evaluator = TartanAirEvaluator()
            results = evaluator.evaluate_one_trajectory(gt_file, poselist, scale=True, kittitype=(datastr=='kitti'))
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

            # save results and visualization
            np.save(osp.join(scene, "average_transform_estimate", "{}_ate.npy".format(fusion_type)), results["ate_score"])
            np.save(osp.join(scene, "average_transform_estimate", "{}_kitti_score.npy".format(fusion_type)), np.array(results["kitti_score"]))
            plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=osp.join(scene, "average_transform_estimate", "{}_traj.png".format(fusion_type)), title='ATE %.4f, KITTI-R/t: %.4f, %.4f' %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
