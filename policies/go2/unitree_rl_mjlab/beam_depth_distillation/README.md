```bash
source src/unitree_lowlevel/scripts/setup.sh eth0 foxy

# Camera
realsense-viewer

ros2 param list /camera/camera

env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 launch realsense2_camera rs_launch.py \
  config_file:="'src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/D435i.yaml'"

# Preprocessor
env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/depth_image_preprocessor.yaml

env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 topic hz /camera/depth/image_rect_raw --window 300
ros2 topic hz /unitree_go2_beam_depth/depth_m --window 300
ros2 topic hz /lowstate --window 300

# Controller
./src/legged_rl_deploy/scripts/run.sh eth0 foxy ros2 run legged_rl_deploy legged_rl_deploy_node eth0 \
    src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/config_explicit.yaml


ros2 bag record -o load-width10-right6 \
  /lowstate \
  /lowcmd \
  /wirelesscontroller \
  /unitree_go2/inertial_estimate \
  /camera/depth/image_rect_raw \
  /camera/depth/camera_info

ros2 bag play load-width10-right6/ --topics /camera/depth/image_rect_raw

python3 src/unitree_lowlevel/scripts/depth_to_video.py load-width10-right6/

tar -I 'zstd -9 -T0' -cf load-width10-right6.tar.zst load-width10-right6/
rsync -avP  unitree@100.88.41.38:/home/unitree/code/unitree_ws/load-width10-right6.tar.zst ~/code/unitree_ws

tar -I zstd -xf ~/code/unitree_ws/load-width10-right6.tar.zst
```