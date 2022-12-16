"""
Script to compute average transforms: Average Quaternion and Average Translation given Mx4 and Mx3 Q and T matrices
"""
import argparse
import os
import os.path as osp
import numpy as np

from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--scene", required=True, type=str)
    return parser.parse_args()


# Tolga Birdal https://github.com/tolgabirdal/averaging_quaternions/blob/master/avg_quaternion_markley.m
def average_quaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = np.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        # handle the antipodal configuration
        if(q[0] < 0):
            q = -q
        # multiply q with its transposed version q' and add A: rank 1 update
        A = np.outer(q, q) + A

    # scale
    A = (1.0 / M) * A
    # compute eigenvalues and vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue: Get the eigenvector corresponding to largest eigen value
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:, 0]).flatten()


def quat2eul(quat_data):
    # Assuming quaternion is [dX, dY, dZ, W]
    quat_data = np.array(quat_data)
    eul = R.from_quat(quat_data).as_euler('xyz', degrees=False)
    return eul

def eul2quat(eul_data):
    eul_data = np.array(eul_data)
    quat = R.from_euler('xyz', angles=eul_data).as_quat()
    return quat

def euler_median_quaternion(Q):
    if len(Q.shape)!=2:
        print("Shape of Q expected to be (N, 4). Given " + str(Q.shape))
    E = np.stack([quat2eul(q) for q in Q], axis=0)  # (N, 3)
    Emedian = np.median(E, axis=0)  # (3,)
    qmedian = eul2quat(Emedian)
    return qmedian

def euler_mean_quaternion(Q):
    if len(Q.shape)!=2:
        print("Shape of Q expected to be (N, 4). Given " + str(Q.shape))
    E = np.stack([quat2eul(q) for q in Q], axis=0)  # (N, 3)
    Emean = np.mean(E, axis=0)  # (3,)
    qmean = eul2quat(Emean)
    return qmean


def average_translations(T):
    return np.mean(T, axis=0)


if __name__ == "__main__":
    args = parse_args()

    if args.version == "mini":
        datastr = "nuscenesmini"
        data_root = '/home/nboloor/slam/data/nuscenes_mini'
    else:
        datastr = "nuscenes"
        data_root = '/home/nboloor/slam/data/nuscenes'

    result_dir = osp.join("results", datastr, args.scene)
    output_dir = osp.join("results", datastr, args.scene, "average_transform_estimate")
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    sensors = [
        'CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
        'CAM_BACK','CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    qs, ts = [], []
    gt_qs, gt_ts = [], []
    for sensor in sensors:
        pred_file = osp.join(result_dir, sensor, "est.npy")
        transform = np.load(pred_file)
        trans = transform[:, :3]
        q = transform[:, 3:]
        ts.append(trans)
        qs.append(q)
        
        # GT quaternion in npy file is already [x, y, z, w] for Nuscenes
        gt_file = osp.join(result_dir, sensor, "gt.npy")
        gt_transform = np.load(gt_file)
        gt_trans = gt_transform[:, :3]
        gt_q = gt_transform[:, 3:]
        gt_qs.append(gt_q)
        gt_ts.append(gt_trans)

    # load the absolute poses across the trajectories
    Q = np.stack(qs, axis=1)  # (N, C, 3)
    T = np.stack(ts, axis=1)  # (N, C, 3) 
    gt_Q = np.stack(gt_qs, axis=1)
    gt_T = np.stack(gt_ts, axis=1)

    N = Q.shape[0]
    avg_rotation = np.zeros((N, 4))
    avg_translation = np.zeros((N, 3))
    median_eul_rotation = np.zeros((N, 4))
    mean_eul_rotation = np.zeros((N, 4))
    gt_avg_rotation = np.zeros((N, 4))
    gt_avg_translation = np.zeros((N, 3))

    # initialization
    avg_rotation[0] = average_quaternions(Q[0])
    median_eul_rotation[0] = euler_median_quaternion(Q[0])
    mean_eul_rotation[0] = euler_mean_quaternion(Q[0])

    # get the averaged GT and translation across cameras
    for i in range(N):
        gt_avg_rotation[i] = average_quaternions(gt_Q[i])
        gt_avg_translation[i] = average_translations(gt_T[i])
        avg_translation[i] = average_translations(T[i])

    # average the relative rotations only
    for i in range(1, N):
        # compute relative R from absolute
        prev_q, curr_q = R.from_quat(Q[i-1]), R.from_quat(Q[i])
        q_rel = (curr_q * prev_q.inv()).as_quat()

        # compute the average relative transformations
        avg_q_rel = R.from_quat(average_quaternions(q_rel))
        eul_med_q_rel = R.from_quat(euler_median_quaternion(q_rel))
        eul_mean_q_rel = R.from_quat(euler_mean_quaternion(q_rel))

        # convert relative transformations back to absolute
        prev_q_abs = R.from_quat(avg_rotation[i-1])
        avg_rotation[i] = (avg_q_rel * prev_q_abs).as_quat()
        median_eul_rotation[i] = (eul_med_q_rel * prev_q_abs).as_quat()
        mean_eul_rotation[i] = (eul_mean_q_rel * prev_q_abs).as_quat()

    transforms = np.concatenate([avg_translation, avg_rotation], axis=1)
    transforms_median = np.concatenate([avg_translation, median_eul_rotation], axis=1)
    transforms_mean = np.concatenate([avg_translation, mean_eul_rotation], axis=1)
    gt_transforms = np.concatenate([gt_avg_translation, gt_avg_rotation], axis=1)
    np.save(osp.join(output_dir, "avg_est.npy"), transforms)
    np.save(osp.join(output_dir, "eul_median_est.npy"), transforms_median)
    np.save(osp.join(output_dir, "eul_mean_est.npy"), transforms_mean)
    np.savetxt(osp.join(output_dir, "avg_gt.txt"), gt_transforms)
