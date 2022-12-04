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
    assert len(Q.shape)==2, f"Shape of Q expected to be (N, 4). Given {Q.shape}"
    E = np.stack([quat2eul(q) for q in Q], axis=0)  # (N, 3)
    Emedian = np.median(E, axis=0)  # (3,)
    qmedian = eul2quat(Emedian)
    return qmedian

def euler_mean_quaternion(Q):
    assert len(Q.shape)==2, f"Shape of Q expected to be (N, 4). Given {Q.shape}"
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
        'CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK','CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    qs, ts = [], []
    for sensor in sensors:
        pred_file = osp.join(result_dir, sensor, "est.npy")
        transform = np.load(pred_file)
        trans = transform[:, :3]
        q = transform[:, 3:]
        ts.append(trans)
        qs.append(q)

    Q = np.stack(qs, axis=1)
    T = np.stack(ts, axis=1)
    N = Q.shape[0]
    avg_rotation = np.zeros((N, 4))
    avg_translation = np.zeros((N, 3))
    median_eul_rotation = np.zeros((N, 4))
    mean_eul_rotation = np.zeros((N, 4))

    for i in range(N):
        avg_rotation[i] = average_quaternions(Q[i])
        avg_translation[i] = average_translations(T[i])
        median_eul_rotation[i] = euler_median_quaternion(Q[i])
        mean_eul_rotation[i] = euler_mean_quaternion(Q[i])

    transforms = np.concatenate([avg_translation, avg_rotation], axis=1)
    transforms_median = np.concatenate([avg_translation, median_eul_rotation], axis=1)
    transforms_mean = np.concatenate([avg_translation, mean_eul_rotation], axis=1)
    np.save(osp.join(output_dir, "avg_est.npy"), transforms)
    np.save(osp.join(output_dir, "eul_median_est.npy"), transforms_median)
    np.save(osp.join(output_dir, "eul_mean_est.npy"), transforms_mean)
