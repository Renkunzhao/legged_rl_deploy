```bash
source src/unitree_lowlevel/scripts/setup.sh eth0 foxy

env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 launch realsense2_camera rs_launch.py \
  config_file:="'src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/D435i.yaml'"

env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 run legged_rl_deploy depth_image_preprocessor_node.py \
  --ros-args --params-file \
  src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/depth_image_preprocessor.yaml

env RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
ros2 topic hz /camera/depth/image_rect_raw --window 300
ros2 topic hz /unitree_go2_beam_depth/depth_m --window 300
ros2 topic hz /lowstate --window 300

./src/legged_rl_deploy/scripts/run.sh eth0 foxy ros2 run legged_rl_deploy legged_rl_deploy_node eth0 \
    src/legged_rl_deploy/policies/go2/unitree_rl_mjlab/beam_depth_distillation/config.yaml

ros2 bag record -o D435i /camera/depth/image_rect_raw /unitree_go2_beam_depth/depth_m /lowstate /lowcmd
tar -I 'zstd -9 -T0' -cf D435i.tar.zst D435i/
rsync -avP  unitree@100.88.41.38:/home/unitree/code/unitree_ws/D435i.tar.zst ~/code/unitree_ws

tar -I zstd -xf ~/code/unitree_ws/D435i.tar.zst

ros2 param list /camera/camera 
  accel_fps
  accel_info_qos
  accel_qos
  align_depth.enable
  align_depth.frames_queue_size
  angular_velocity_cov
  base_frame_id
  camera_name
  clip_distance
  color_info_qos
  color_qos
  colorizer.color_scheme
  colorizer.enable
  colorizer.frames_queue_size
  colorizer.histogram_equalization_enabled
  colorizer.max_distance
  colorizer.min_distance
  colorizer.stream_filter
  colorizer.stream_format_filter
  colorizer.stream_index_filter
  colorizer.visual_preset
  decimation_filter.enable
  decimation_filter.filter_magnitude
  decimation_filter.frames_queue_size
  decimation_filter.stream_filter
  decimation_filter.stream_format_filter
  decimation_filter.stream_index_filter
  depth_info_qos
  depth_module.auto_exposure_roi.bottom
  depth_module.auto_exposure_roi.left
  depth_module.auto_exposure_roi.right
  depth_module.auto_exposure_roi.top
  depth_module.emitter_always_on
  depth_module.emitter_enabled
  depth_module.emitter_on_off
  depth_module.enable_auto_exposure
  depth_module.error_polling_enabled
  depth_module.exposure
  depth_module.frames_queue_size
  depth_module.gain
  depth_module.global_time_enabled
  depth_module.hdr_enabled
  depth_module.inter_cam_sync_mode
  depth_module.laser_power
  depth_module.output_trigger_enabled
  depth_module.profile
  depth_module.sequence_id
  depth_module.sequence_name
  depth_module.sequence_size
  depth_module.visual_preset
  depth_qos
  device_type
  diagnostics_period
  disparity_filter.enable
  disparity_to_depth.enable
  enable_accel
  enable_color
  enable_depth
  enable_gyro
  enable_infra1
  enable_infra2
  enable_sync
  filter_by_sequence_id.enable
  filter_by_sequence_id.frames_queue_size
  filter_by_sequence_id.sequence_id
  gyro_fps
  gyro_info_qos
  gyro_qos
  hdr_merge.enable
  hdr_merge.frames_queue_size
  hold_back_imu_for_frames
  hole_filling_filter.enable
  hole_filling_filter.frames_queue_size
  hole_filling_filter.holes_fill
  hole_filling_filter.stream_filter
  hole_filling_filter.stream_format_filter
  hole_filling_filter.stream_index_filter
  infra1_info_qos
  infra1_qos
  infra2_info_qos
  infra2_qos
  initial_reset
  json_file_path
  linear_accel_cov
  motion_module.enable_motion_correction
  motion_module.frames_queue_size
  motion_module.global_time_enabled
  pointcloud.allow_no_texture_points
  pointcloud.enable
  pointcloud.filter_magnitude
  pointcloud.frames_queue_size
  pointcloud.ordered_pc
  pointcloud.pointcloud_qos
  pointcloud.stream_filter
  pointcloud.stream_format_filter
  pointcloud.stream_index_filter
  publish_odom_tf
  publish_tf
  reconnect_timeout
  rgb_camera.auto_exposure_priority
  rgb_camera.auto_exposure_roi.bottom
  rgb_camera.auto_exposure_roi.left
  rgb_camera.auto_exposure_roi.right
  rgb_camera.auto_exposure_roi.top
  rgb_camera.backlight_compensation
  rgb_camera.brightness
  rgb_camera.contrast
  rgb_camera.enable_auto_exposure
  rgb_camera.enable_auto_white_balance
  rgb_camera.exposure
  rgb_camera.frames_queue_size
  rgb_camera.gain
  rgb_camera.gamma
  rgb_camera.global_time_enabled
  rgb_camera.hue
  rgb_camera.power_line_frequency
  rgb_camera.profile
  rgb_camera.saturation
  rgb_camera.sharpness
  rgb_camera.white_balance
  rosbag_filename
  serial_no
  spatial_filter.enable
  spatial_filter.filter_magnitude
  spatial_filter.filter_smooth_alpha
  spatial_filter.filter_smooth_delta
  spatial_filter.frames_queue_size
  spatial_filter.holes_fill
  spatial_filter.stream_filter
  spatial_filter.stream_format_filter
  spatial_filter.stream_index_filter
  temporal_filter.enable
  temporal_filter.filter_smooth_alpha
  temporal_filter.filter_smooth_delta
  temporal_filter.frames_queue_size
  temporal_filter.holes_fill
  temporal_filter.stream_filter
  temporal_filter.stream_format_filter
  temporal_filter.stream_index_filter
  tf_publish_rate
  unite_imu_method
  usb_port_id
  use_sim_time
  wait_for_device_timeout

```
