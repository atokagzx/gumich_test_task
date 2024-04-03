import pyrealsense2 as rs
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector
import cv2
import numpy as np
import logging


class RSWrapper:
    def __init__(self):
        self._logger = logging.getLogger("rs_wrapper")
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)
        self._config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._profile = self._pipeline.start(self._config)
        self._depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = self._depth_sensor.get_depth_scale()
        self._align = rs.align(rs.stream.color)

    
    def iterate_over_frames(self):
        while True:
            frames = self._pipeline.wait_for_frames()
            aligned_frames = self._align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                self._logger.warning("frames are empty")
                continue
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)
            vtx = np.asarray(points.get_vertices())
            tex = np.asarray(color_frame.get_data())
            pcd = PointCloud()
            pcd.points = Vector3dVector(vtx)
            pcd.colors = Vector3dVector(tex / 255)  # Normalize colors to [0, 1]


    def _create_point_cloud(self, depth_frame):
        depth_image = np.asanyarray(depth_frame.get_data())
        colorized_depth = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_RAINBOW)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        pcd = PointCloud()
        pcd.points = Vector3dVector(depth_intrinsics.deproject(depth_image))
        return pcd