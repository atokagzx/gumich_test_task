import open3d as o3d
import json

from open3d.visualization import VisualizerWithKeyCallback, ViewControl
from open3d.geometry import PointCloud, Image, RGBDImage
from open3d.utility import Vector3dVector
from open3d.camera import PinholeCameraIntrinsic

print(o3d.t.io.RealSenseSensor.list_devices())

config ={
    "serial": "f1270667",
    "color_format": "RS2_FORMAT_RGB8",
    "color_resolution": "640,480",
    "depth_format": "RS2_FORMAT_Z16",
    "depth_resolution": "320,240",
    "fps": "30",
    "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE"
 }

bag_filename = "test.bag"
rs_cfg = o3d.t.io.RealSenseSensorConfig(config)

rs = o3d.t.io.RealSenseSensor()
rs.init_sensor(rs_cfg, 0, bag_filename)
intrinsics = rs.get_metadata().intrinsics
rs.start_capture(True)
vis = VisualizerWithKeyCallback()
vis.create_window()
vis.get_view_control().change_field_of_view(step=-1)

while True:
    im_rgbd = rs.capture_frame(True, True) 
    depth = o3d.geometry.Image(im_rgbd.depth.cpu())
    color = o3d.geometry.Image(im_rgbd.color)
    im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsics)
    
    vis.clear_geometries()
    vis.add_geometry(pcd)
    # draw camera tf
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    camera.translate([0, 0, 0])
    vis.add_geometry(camera)

    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
