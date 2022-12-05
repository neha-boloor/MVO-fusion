# nuscenesmini
for SCENE_ID in 0061 0103 0553 0655 0757 0796 0916 1077 1094 1100 
  do
    for CAM in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT
    do
    python vo_trajectory_from_folder.py --model-name tartanvo_1914.pkl --nuscenesmini --batch-size 1 --worker-num 0 --test-dir ../data/nuscenes_mini/scenes/scene-$SCENE_ID/$CAM/key_frames --pose-file ../data/nuscenes_mini/scenes/scene-$SCENE_ID/$CAM/pose_files/pose.txt
    done
done
# nuscenes
for SCENE_ID in $(seq -f "%04g" 1 36) $(seq -f "%04g" 38 39) $(seq -f "%04g" 41 76) $(seq -f "%04g" 92 102) 
  do
    for CAM in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT
    do
    python vo_trajectory_from_folder.py --model-name tartanvo_1914.pkl --nuscenes --batch-size 1 --worker-num 0 --test-dir ../data/nuscenes/scenes/scene-$SCENE_ID/$CAM/key_frames --pose-file ../data/nuscenes/scenes/scene-$SCENE_ID/$CAM/pose_files/pose.txt
    done
done
# Sample command:
# > python vo_trajectory_from_folder.py  --model-name tartanvo_1914.pkl --nuscenesmini --batch-size 1 --worker-num 0 --test-dir ../data/nuscenes_mini/scenes/scene-0061/CAM_FRONT_LEFT/key_frames --pose-file ../data/nuscenes_mini/scenes/scene-0061/CAM_FRONT_LEFT/pose_files/pose.txt
