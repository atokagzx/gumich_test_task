#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import logging
from utils import RSWrapper, CLIPSegAdapter, SAMAdapter, Segment3D
import argparse
import json
import cv2

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("main")
    argparser = argparse.ArgumentParser()
    argparser.add_argument('config',
                        type=str,
                        help='Path to the config file')
    configs = argparser.parse_args()
    with open(configs.config) as f:
        rs_config = json.load(f)

    clipseg = CLIPSegAdapter()
    sam = SAMAdapter()
    rs_wrapper = RSWrapper(rs_config)
    for color, depth in rs_wrapper.iterate_over_frames():
        mask, ids = clipseg.segment(color,'object', 0.1)
        clipseg_mask = np.where(mask < 50, 0, 255).astype(np.uint8)
        sam_mask = sam.segment(color, clipseg_mask)
        masked_pcd = rs_wrapper.rgbd_to_pointcloud(color, depth, sam_mask)
        original_pcd = rs_wrapper.rgbd_to_pointcloud(color, depth)

        results = Segment3D.find_normals(masked_pcd)
        results_visual = []
        for result in results:
            results_visual.append(result.plane_cloud)
            results_visual.append(result.plane_tf)
            results_visual.append(result.bbox)
        logger.info(f"found {len(results)} planes")
        for i, result in enumerate(results):
            logger.info(f'picking point "{i}" TF:\n{result.tf_matrix}\ndims: {result.dims}')
        o3d.visualization.draw_geometries([original_pcd] + results_visual)
        