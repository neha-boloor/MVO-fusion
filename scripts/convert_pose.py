"""
Script to generate pose files per camera in the nuscenes dataset
"""
import argparse
import os
import os.path as osp
import shutil

import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.version = "v1.0-" + args.version
    if args.version == "v1.0-mini":
        data_root = '/home/nboloor/slam/data/nuscenes_mini'
    else:
        data_root = '/home/nboloor/slam/data/nuscenes'
    nusc = NuScenes(version=args.version, dataroot=data_root, verbose=True)
    sensors = [
        'CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK','CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]

    '''
    Fetch the scene, the first and last sample in that scene
    '''
    for scene in tqdm(nusc.scene):
        # print('first sample token in the scene: ', scene['first_sample_token'])
        start_sample = nusc.get('sample', scene['first_sample_token'])
        last_sample = nusc.get('sample', scene['last_sample_token'])

        '''
        For each sensor [6 cameras in our case] get ego_key_token of key frames only
        '''
        for sensor in sensors:
            data_scene_dir = osp.join(data_root, "scenes", scene['name'], sensor)
            key_frame_dir = osp.join(data_scene_dir, "key_frames")
            pose_dir = osp.join(data_scene_dir, "pose_files")
            intrinsic_dir = osp.join(data_scene_dir, "intrinsics")
            if not osp.exists(key_frame_dir):
                os.makedirs(key_frame_dir)
            if not osp.exists(pose_dir):
                os.makedirs(pose_dir)
            if not osp.exists(intrinsic_dir):
                os.makedirs(intrinsic_dir)

            start_cam = nusc.get('sample_data', start_sample['data'][sensor])
            last_cam = nusc.get('sample_data', last_sample['data'][sensor])

            # store the intrinsincs of the camera
            calibration_token = start_cam['calibrated_sensor_token']
            calibration = nusc.get('calibrated_sensor', calibration_token)["camera_intrinsic"]
            fx, fy = calibration[0][0], calibration[1][1]
            cx, cy = calibration[0][2], calibration[1][2]
            intrinsic_file = osp.join(intrinsic_dir, "intrinsics.npy")
            instrinsics = np.array([fx, fy, cx, cy])
            np.save(intrinsic_file, instrinsics)

            frame_count = 0
            key_frame_sample_tokens, ego_key_tokens = [], []
            if start_cam['is_key_frame']:
                frame_count += 1

                # copy image to keyframe directory
                keyframe_file = osp.join(data_root, start_cam['filename'])
                shutil.copy2(keyframe_file, key_frame_dir)

                key_frame_sample_tokens.append(start_cam['sample_token'])
                ego_key_tokens.append(start_cam['ego_pose_token'])
            while not start_cam['next'] == "":
                next_cam = nusc.get('sample_data', start_cam['next'])
                while not next_cam['is_key_frame']:
                    next_cam = nusc.get('sample_data', next_cam['next'])
                frame_count += 1
                start_cam = next_cam
                key_frame_sample_tokens.append(start_cam['sample_token'])
                ego_key_tokens.append(start_cam['ego_pose_token'])

                # copy image to keyframe directory
                keyframe_file = osp.join(data_root, start_cam['filename'])
                shutil.copy2(keyframe_file, key_frame_dir)

            '''
            Store the  poses per camera in a txt file
            '''
            pose_file = osp.join(pose_dir, "pose.txt")
            with open(pose_file, 'w') as log:
                for ego in ego_key_tokens:
                    ego_ele = nusc.get('ego_pose', ego)
                    string = ''
                    for ele in ego_ele['translation']:
                        string += str(ele) + ' '
                    for ele in ego_ele['rotation']:
                        string+=str(ele) +  ' '
                    string += '\n'
                    log.write(string)
