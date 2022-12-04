"""Example usage of Pytorch Boiler"""
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, './')
sys.path.insert(0, '../')
sys.path.insert(0, '../pytorch_boiler')

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F

from average_transforms import average_quaternions, average_translations
from fusion_module import FusionModule
from nuscenes_loader import NuScenes, DataLoader
from pytorch_boiler import Boiler, overload
from TartanVO import TartanVO as TartanVOModel


def quat2so(quat_data):
    quat_data = np.array(quat_data)
    sin_half_theta = np.sqrt(np.sum(quat_data[0:3] * quat_data[0:3]))
    axis = quat_data[0:3] / sin_half_theta
    cos_half_theta = quat_data[3]
    theta = 2 * np.arctan2(sin_half_theta, cos_half_theta)
    so = theta * axis
    return so


class Trainer(Boiler):

    @overload
    def pre_process(self, batch):
        data = []
        for i in range(len(batch['CAM_FRONT']['image'])):    
            sample_per_camera = {}
            for camera in NuScenes.sensors:
                cam = batch[camera]
                sample_per_camera[camera] = {
                    'img': cam['image'][i][None, ...].cuda(),
                    'rot': cam['rotation'][i][None, ...],
                    'trans': cam['translation'][i][None, ...],
                    'intrins': cam['intrinsic'][i][None, ...].cuda()
                }
            data.append(sample_per_camera)
        return data

    @overload
    def loss(self, model_output, batch):  # Overload the loss function
        data = self.pre_process(batch)
        
        # model_output = (39, 6)
        rot = model_output[:, :3]
        trans = model_output[:, 3:]

        gt_trans = []
        gt_rot = []
        for i in range(1, len(data)):
            rel_rot_quat, rel_trans = [], []
            for camera in data[i]:
                # get the relative rotation for GT
                r_curr = R.from_quat(data[i][camera]['rot'].numpy())
                r_prev = R.from_quat(data[i-1][camera]['rot'].numpy())
                r_rel = r_curr * r_prev.inv()
                rel_quat = torch.from_numpy(r_rel.as_quat()).cuda()
                rel_rot_quat.append(rel_quat)

                # get the relative translation for GT
                t_curr = data[i][camera]['trans']
                t_prev = data[i-1][camera]['trans']
                rel_trans.append(t_curr - t_prev)

            rel_rot_quat = torch.cat(rel_rot_quat, dim=0)
            rel_trans = torch.cat(rel_trans, dim=0)
            
            gt_rot.append(
                torch.from_numpy(
                    quat2so(
                        average_quaternions(rel_rot_quat.cpu().detach().numpy())
                    )
                ).float()
            )
            gt_trans.append(
                torch.from_numpy(
                    average_translations(rel_trans.cpu().detach().numpy())
                ).float()
            )
        gt_trans = torch.stack(gt_trans, dim=0).cuda()  # (39, 3)
        gt_rot = torch.stack(gt_rot, dim=0).cuda()  # (39, 3)

        trans_loss = F.cosine_embedding_loss(trans.cuda(), gt_trans.cuda(), target=torch.FloatTensor([1]).cuda(), reduction='mean')
        rot_loss = F.mse_loss(rot.cuda(), gt_rot.cuda(), reduction='mean')

        total_loss = trans_loss + rot_loss

        return {'rot_loss': rot_loss, 'trans_loss': trans_loss, 'summary': total_loss}  # Can return a tensor, or a dictinoary like {'xe_loss': xe_loss} with multiple losses. See README.

    # @overload
    # def performance(self, model_output, data):
    #     images, labels = data
    #     preds = model_output.argmax(dim=-1)
    #     acc = (preds == labels.cuda()).float().mean()
    #     return acc.cpu().detach().numpy()  # Can return a tensor, or a dictinoary like {'acc': acc} with multiple metrics. See README.


if __name__ == '__main__':
    config = {
                'exp_tag': 'exp1_full_nuscenes',
                'batch_size': 40,
                'epochs': 500,
                'input_dim': 256*6*6,
                'latent_dim': 512,
                'num_layers': 4
            }
    # nuscenes_dataset = NuScenes(dataroot='/home/nboloor/slam/data/nuscenes_mini', version='v1.0-mini')  # Nuscenes mini
    nuscenes_dataset = NuScenes(dataroot='/home/nboloor/slam/data/nuscenes')  # Nuscenes full
    train_dataloader = DataLoader(nuscenes_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=0)
    val_dataloader = DataLoader(nuscenes_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=False,
                                  num_workers=0)

    tartanvo_model = TartanVOModel(model_name='tartanvo_1914.pkl', root_path='../../models', return_feats=True)
    fusion_module = FusionModule(tartanvo_model=tartanvo_model, 
                                 input_dim=config['input_dim'],
                                 latent_dim=config['latent_dim'],
                                 num_layers=3).cuda()
    optimizer = torch.optim.Adam(params=fusion_module.parameters(), lr=5e-4, weight_decay=1e-5)

    trainer = Trainer(model=fusion_module, optimizer=optimizer, 
                      train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                      epochs=config['epochs'], save_path=f'./model/{config["exp_tag"]}/state_dict.pt', load_path=None, mixed_precision=False)
    trainer.fit()
