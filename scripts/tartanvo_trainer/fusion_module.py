from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class FusionModule(nn.Module):

    def __init__(self, tartanvo_model, input_dim=256*6*6, latent_dim=256, num_layers=3):
        super(FusionModule, self).__init__()

        self.input_dim = input_dim  # 256 x 6 (see fcnum variable inside )
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.tartanvo_model = tartanvo_model  # Must implement
        self.tartanvo_model.eval()

        layers = [nn.Linear(input_dim, latent_dim), nn.ELU()]
        for i in range(num_layers):
            layers.extend(
                [nn.Linear(latent_dim//(2**i), latent_dim//(2**(i+1))), nn.ELU()]
            )
        layers.append(nn.Linear(latent_dim//(2**num_layers), 6))
        self.model = nn.Sequential(*layers)  # input -> (N, 256*6), output -> (N, 6) : (rot, trans)

    def forward_per_sample(self, sample_per_camera):

        '''
        img0   = sample['img1'].cuda()
        img1   = sample['img2'].cuda()
        intrinsic = sample['intrinsic'].cuda()
        inputs = [img0, img1, intrinsic]
        '''
        self.tartanvo_model.eval()
        with torch.no_grad():
            feats = []
            for camera in sample_per_camera:
                sample = sample_per_camera[camera]
                _, _, featsnp = self.tartanvo_model.test_batch(sample)
                feats.append(torch.from_numpy(featsnp).cuda())
            feats = torch.cat(feats, dim=-1)  # (N, C*d)
        assert len(feats.shape) == 2  # (N, C*d)

        output_rot_trans = self.model(feats)
        return output_rot_trans

    def forward(self, data):
        # data = [
        #   {'CAM_FRONT': {'img', ..}, 
        #    'CAM_BACK': {...}
        #    ...
        #   },
        #
        #   {'CAM_FRONT': {'img', ..}, 
        #    'CAM_BACK': {...}
        #    ...
        #   },
        # 
        #   ... 40 samples
        # ]
        N = len(data)
        outputs = []
        for i in tqdm(range(1, N)):
            data_per_camera_1 = data[i-1]
            data_per_camera_2 = data[i]

            sample_per_camera = {}
            for camera in data_per_camera_1.keys():
                # find the motion between the two cameras
                r_curr = R.from_quat(data_per_camera_2[camera]['rot'].numpy())
                r_prev = R.from_quat(data_per_camera_1[camera]['rot'].numpy())
                r_rel = r_curr * r_prev.inv()
                rel_quat = torch.from_numpy(r_rel.as_quat())

                t_curr = data_per_camera_2[camera]['trans']
                t_prev = data_per_camera_1[camera]['trans']
                rel_trans = t_curr - t_prev
                rel_motion = torch.cat([rel_trans, rel_quat], dim=1)

                sample_per_camera[camera] = {
                        'img1': data_per_camera_1[camera]['img'],
                        'img2': data_per_camera_2[camera]['img'],
                        'intrinsic': data_per_camera_2[camera]['intrins'],
                        'motion': rel_motion
                }
            out = self.forward_per_sample(sample_per_camera)
            outputs.append(out)  # (N-1, 6)
        outputs = torch.cat(outputs, dim=0)  # (N-1, 6)
        return outputs
