import numpy as np
import open3d as o3d
import logging
from utils import RSWrapper, CLIPSegAdapter, Segment3D
import argparse
import json

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
    rs_wrapper = RSWrapper(rs_config)
    for color, depth in rs_wrapper.iterate_over_frames():
        mask, ids = clipseg.segment(color,'object', 0.1)
        mask = np.where(mask < 30, 0, 255).astype(np.uint8)
        pcd = rs_wrapper.rgbd_to_pointcloud(color, depth, mask)
        results = Segment3D.find_normals(pcd)
        results_visual = []
        for result in results:
            results_visual.append(result.plane_cloud)
            results_visual.append(result.plane_tf)
            results_visual.append(result.bbox)
        logger.info(f"found {len(results)} planes")
        for i, result in enumerate(results):
            logger.info(f'picking point "{i}" TF:\n{result.tf_matrix}\ndims: {result.dims}')
        o3d.visualization.draw_geometries([pcd] + results_visual)
        