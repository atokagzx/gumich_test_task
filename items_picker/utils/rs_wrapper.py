import open3d as o3d
import cv2
import numpy as np
import logging


class RSWrapper:
    def __init__(self, rs_config):
        self._logger = logging.getLogger("rs_wrapper")
        self._rs_cfg = o3d.t.io.RealSenseSensorConfig(rs_config)

        self._rs = o3d.t.io.RealSenseSensor()
        self._rs.init_sensor(self._rs_cfg, 0)
        self._intrinsics = self._rs.get_metadata().intrinsics
        self._rs.start_capture(True)

    
    def iterate_over_frames(self):
        while True:
            im_rgbd = self._rs.capture_frame(True, True) 
            depth = o3d.geometry.Image(im_rgbd.depth.cpu())
            color = o3d.geometry.Image(im_rgbd.color)
            yield color, depth


    def rgbd_to_pointcloud(self, color, depth, mask):
        # apply mask to depth
        depth = np.array(depth)
        print(depth.shape, mask.shape)
        depth = np.where(mask == 0, 0, depth)
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color)
        im_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(im_rgbd, self._intrinsics)
        return pcd
    

if __name__ == '__main__':
    rs_config ={
        "serial": "f1270667",
        "color_format": "RS2_FORMAT_RGB8",
        "color_resolution": "640,480",
        "depth_format": "RS2_FORMAT_Z16",
        "depth_resolution": "320,240",
        "fps": "30",
        "visual_preset": "RS2_L500_VISUAL_PRESET_SHORT_RANGE"
    }
    rs_wrapper = RSWrapper(rs_config)
    for color, depth in rs_wrapper.iterate_over_frames():
        pcd = rs_wrapper.rgbd_to_pointcloud(color, depth)
        o3d.visualization.draw_geometries([pcd])
