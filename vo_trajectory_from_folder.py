from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, dataset_intrinsics, DownscaleFlow, plot_traj, visflow
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from Datasets.transformation import ses2poses_quat
from evaluator.tartanair_evaluator import TartanAirEvaluator
from TartanVO import TartanVO

import argparse
import numpy as np
import cv2
from os import makedirs
from os.path import isdir
import os.path as osp

def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--nuscenesmini', action='store_true', default=False,
                        help='nuscenes-mini test (default: False)')
    parser.add_argument('--nuscenes', action='store_true', default=False,
                        help='nuscenes test (default: False)')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--save-features', action='store_true', default=False,
                        help='save layer 5 features from TartanVO (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    testvo = TartanVO(args.model_name, return_feats=args.save_features)

    # load trajectory data from a folder
    intrinsic_file = None
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    elif args.nuscenesmini:
        datastr = 'nuscenesmini'
        intrinsic_file = args.pose_file.replace(
            "pose_files", "intrinsics").replace("pose.txt", "intrinsics.npy")
    elif args.nuscenes:
        datastr = 'nuscenes'
        intrinsic_file = args.pose_file.replace(
            "pose_files", "intrinsics").replace("pose.txt", "intrinsics.npy")
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr, intrinsic_file=intrinsic_file) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr=='kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir,  posefile = args.pose_file, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    motionlist = []
    testname = datastr + '_' + args.model_name.split('.')[0]
    camera = args.test_dir.split('/')[-2]
    scene = args.test_dir.split('/')[-3]
    if args.save_flow:
        flowdir = 'results/' + testname + '_flow'
        if not isdir(flowdir):
            makedirs(flowdir)
        flowcount = 0
    if args.save_features:
        featdir = 'results/' + testname + '_feat'
        if not isdir(featdir):
            makedirs(featdir)
        featcount = 0
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break

        if args.save_features:
            motions, flow, feats = testvo.test_batch(sample)
        else:
            motions, flow = testvo.test_batch(sample)
        motionlist.extend(motions)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1
        
        if args.save_features:
            for k in range(feats.shape[0]):
                featk = feats[k]
                np.save(featdir+'/' + camera + '/' +str(featcount).zfill(6)+'.npy',featk)
                featcount += 1

    poselist = ses2poses_quat(np.array(motionlist))

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(args.pose_file, poselist, scale=True, kittitype=(datastr=='kitti'))
        if datastr=='euroc':
            print("==> ATE: %.4f" %(results['ate_score']))
        else:
            print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        result_dir = osp.join("results", datastr, scene, camera)
        if not isdir(result_dir):
            makedirs(result_dir)
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname=result_dir+'/trajectory.png', title='ATE: %.4f,\t KITTI-R/t: %.4f, %.4f' %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))
        np.save(osp.join(result_dir, "est.npy"), results["est_aligned"])
        np.save(osp.join(result_dir, "gt.npy"), results["gt_aligned"])
        np.save(osp.join(result_dir, "ate.npy"), results["ate_score"])
        np.save(osp.join(result_dir, "kitti_score.npy"), np.array(results["kitti_score"]))
        np.save(osp.join(result_dir, "poselist.npy"), np.array(poselist))
    else:
        # plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        pass
    # np.save('results/'+testname+'.npy',poselist)