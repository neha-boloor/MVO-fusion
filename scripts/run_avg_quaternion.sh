cd /tartanvo
# Run average transforms
# nuscenesmini
for SCENE_ID in 0061 0103 0553 0655 0757 0796 0916 1077 1094 1100 
  do
    python scripts/average_transforms.py --scene scene-$SCENE_ID --version mini
done
# nuscenes
for SCENE_ID in $(seq -f "%04g" 1 76) $(seq -f "%04g" 92 102) 
  do
  do
    python scripts/average_transforms.py --scene scene-$SCENE_ID --version trainval
done
# python scripts/average_transforms.py --scene scene-0061 --version mini