import numpy as np
import cv2
from utils import RSWrapper
from open3d.visualization import Visualizer
from open3d.geometry import PointCloud
from open3d.utility import Vector3dVector

if __name__ == '__main__':
    rs_wrapper = RSWrapper()
    vis = Visualizer()
    vis.create_window()
    for pcd, color_image in rs_wrapper.iterate_over_frames():
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        cv2.imshow("color", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
