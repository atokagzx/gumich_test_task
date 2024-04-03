import pyrealsense2 as rs
import numpy as np
import logging


class RSWrapper:
    def __init__(self):
        self._logger = logging.getLogger("rs_wrapper")
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            yield depth_image, color_image
            