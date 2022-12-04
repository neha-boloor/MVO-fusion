# MVO-Fusion
This repository currently runs TartanVO-inference on Key-frames from the 6 cameras from Nuscenes-Mini and Nuscenes Dataset on a Scene-by-Scene basis and plots the estimated trajectories for each of them on a 2D plot.
## Set-up
1. Follow the steps as given in [TartanVO](https://github.com/neha-boloor/MVO-fusion/blob/main/README_TartanVO.md) to set up the docker. 
2. Note the additional steps given below when you bring up the container.

### Additional steps to be run from within the docker container :
#### Install unzip
```bash
apt-get install zip unzip
```
#### Command to run docker from the tartanvo directory:
```bash
nvidia-docker run -it -d --rm --network host --ipc=host -v $PWD:/tartanvo amigoshan/tartanvo:latest
```
#### Command to bring up the container everytime :
```bash
docker exec -it <image_id> bash
Ex: docker exec -it aa492d8e86a5 bash
```

## Download Nuscenes-mini / Nuscenes and copy to the data folder within the container

### Install nuscenes-devkit
```bash
pip install nuscenes-devkit
```
### Download Nuscenes Dataset into the data folder
```bash
wget "link+accesskey" -O nuscenes // DOWNLOAD BOTH THE DATA AND METADATA
Ex:  wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval01_blobs.tgz?AWSAccessKeyId=ASIA6RIK4RRMDGWCESWV&Signature=c7erVRSnDKpbPPV9SoEOc3XE%2FFU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHsaCXVzLWVhc3QtMSJGMEQCIBQMXdNobzxOnUV%2BQu5VeaATsOYoBZdqHoCsR0HJ6faLAiAAuESd1uZzr27no9h1sNCvUOivcd%2BdnMA9sNRfoNkFvCr9AgiU%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAMaDDk5OTEzOTYwOTY4OCIMM5G1%2FTfPEu6pgGxDKtECRfiq6Yn712jeUl1CuZ3RUI1cYSkmKRZLcMKHjPQBuZI2rrCoBpClb8uMwLEbprOFY%2B5%2FlbeIv6sNAPLgkD4VedOwgCxl6rplPFEYo7NCUWScaaAH1q4wcHzWKpsF32gJdq88d2P9v%2FoS8YiCRtRzv42ub%2B2h6KrYxN3RLp0HLuUekWbto2P3FRMfB9TYmqudQUVYCcbxw2cv9mhrQ8YwqZvCfU4u1qkCCOVZTeuuoUXs4RDAHxi2RF96ioSQYh8AfH4D0U%2FfrXAJYKALRdTy3j6AqqHOfJRD%2B6AOoO8Ge2e4MuDD2dMUQ1LjEFlHDCnUGSnGGf5DYGQ7gXsJuFBv%2BPsYPUkiXDC3B2eWAvu3spOu4cklmi21r%2BOkEvIcq9UrxwigKszwHQTAyXhsHTXGb9865fCkyAuMdWexFcaOtocZhPDwtdn%2Fn8cgkbYljqlDIzD58qOcBjqfAVTdBd7HDk2gLGC%2FVwMhIJKRs29gPUvfYKwzC%2Fskye0QAEZkdjTX6NBlLWuljLfEsFTVd1%2BiQFYtKediclSwgJxF85WNVD2pVm0GVtKpvj4VCiyiUCtAbZmlEjCM68LUvY2cLCGv20YbAEBoWRUh%2Bqdw9Rep2aqydwSt%2FwOZsKt4E2u7ouVB6C6mi1CZgVaMCsfkKcn0ApXARz5vqPGq4A%3D%3D&Expires=1670353147" -O nuscenes
wget "https://s3.amazonaws.com/data.nuscenes.org/public/v1.0/v1.0-trainval_meta.tgz?AWSAccessKeyId=ASIA6RIK4RRMAPSPTGWP&Signature=%2F%2FnHApnatj6ZV7SFofyHj0jxyY8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHwaCXVzLWVhc3QtMSJHMEUCIQDqe7P34Le5WX316x5G2byqQfxqHz59rp%2BK1M%2BGwTqiHgIgGJV7StCQE5oK8LRNM3%2F31twoCFsyzm9nbyi4ghi95oQq%2FQIIlf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARADGgw5OTkxMzk2MDk2ODgiDBDiMbEqxX4sIzjBYSrRAijT%2BQnEizLA22iU%2BNTUWUNFh%2FDdRYFRibQVFb4djOWCQGpifHYh9rVvyNgtgyJBHC9CMHzK7%2BXTSv3ezYLe1dnYmJ3at%2FGqyJSITHHGu51KuHxk9PqoLtUNeeVooosjVDdOBP40QCTnstzLm47j%2BTD9CedoDLixUXZxC18SaH3VxlGj1Nf7FN61ZuLgRtozuSWNN%2FWNh0zJ2l%2BZnAJ4NcMMGAeXAoleK4e2DDrHdbL4FSu9L6QchfZVrweRW9VYgpSLL87hMY5fPmqTymoDodhRvBzmSiLsIB7F1KjykierNNc5yPLGHDsacu%2FJ5OBsJ0tNqXv8gxbKoG0PDbcypXeJVLph6fJ%2F4NBSwIALzKfAadImp089n4SvP3fYC2J2SCejTuKpBYS4%2BtQXpyJ7NazPOVY0qOkKwo8mqQF%2F1Y3oFz9r51O%2Fp1IwuoHBfk4W2dkwroiknAY6ngEwTERDmtjXHtHVmBTr6S9%2FD4Gcpf7SVDpAPVKU9eWaYzWqqT%2Bg7F5XtRoODM3cyYLCMS%2B3PrlkmVnC67lsjJmkrGI2z4br2vqRuWuVC2XAOPRUPSPxjhdoMB1tT9XMDgjm2Cs2coee0aoRtzN%2FwObtKD5ie03%2F2rvZVih2GyBt3qQ4btrwAYsuh4Rva2myjeGVtN5P5BCpg3n9Fa%2Ba8A%3D%3D&Expires=1670356272" -o v1.0-trainval

wget https://www.nuscenes.org/data/v1.0-mini.tgz -O nuscenes-mini
```

## Model Inference

### Command to run inference per dataset per scene per camera

```bash
python vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl --nuscenesmini --batch-size 1 --worker-num 0 --test-dir ../data/nuscenes_mini/scenes/scene-0061/CAM_BACK/key_frames --pose-file ../data/nuscenes_mini/scenes/scene-0061/CAM_BACK/pose_files/pose.txt
```

### Command to run inference on all datasets [nuscenesmini and nuscenes] on all scenes and all 6 cameras:
```bash
sh scripts/run_inference.sh
```

* The results are stored on the results folder: '/results/<dataset_name>/<scene_name>/<sensor_name>/
* The results include the GT and Estimated trajectories, the ATE score for the 2D trajectory plotted for the same.
* Ex: Output for nuscenesmini, scene-0061, for CAM_FRONT can be found at :
    The directory structure looks like ![this](./figures/results_dir.png)

## Fusion Approach #1: [Average Rotation + Translation]
We use the approach described in this [paper](http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf) to perform a quaternion average for rotation and vector mean for translation to fuse the predictions from 6 sensors.
### Command to run quaternion + translation average:

* Per dataset per scene:
```bash
python scripts/average_transforms.py --scene scene-0061 --version mini
```

* Across all datasets and scenes:
```bash
sh scripts/run_avg_quaternion.sh
```
This stores the average estimate under the results/<dataset_name>/<scene_name>/average_transform_estimate directory.
