import open3d as o3d
from open3d.geometry import PointCloud
import logging
import numpy as np
import typing
from dataclasses import dataclass


@dataclass
class SegmentResult:
    plane_cloud: PointCloud
    plane_tf: o3d.geometry.TriangleMesh
    bbox: o3d.geometry.OrientedBoundingBox
    dims: typing.Tuple[float, float]
    tf_matrix: np.ndarray

class Segment3D:
    _logger = logging.getLogger("segment3d")
    
    @staticmethod
    def validate_rotation_matrix(R):
        """
        Check if a matrix is a valid rotation matrix.
        """
        is_orthogonal = np.allclose(np.dot(R, R.T), np.eye(3))
        is_det_one = np.isclose(np.linalg.det(R), 1)
        return is_orthogonal and is_det_one


    @classmethod
    def find_rotation_matrix(cls, plane_eq):
        '''
        Compute the rotation matrix for a plane given by its equation Ax + By + Cz + D = 0
        @param plane_eq: tuple of 4 floats (A, B, C, D)
        @return: 3x3 numpy array
        '''
        A, B, C, _D = plane_eq
        norm = np.sqrt(A**2 + B**2 + C**2)
        A /= norm
        B /= norm
        C /= norm
        N = np.array([A, B, C])
        Z = np.array([0, 0, 1])
        V = np.cross(N, Z)
        V /= np.linalg.norm(V)
        U = np.cross(V, N)
        R = np.column_stack((N, U, V))
        assert Segment3D.validate_rotation_matrix(R)
        return R


    @classmethod
    def find_normals(cls, pcd: PointCloud) -> typing.List[SegmentResult]:
        '''
        Segment the point cloud into clusters using DBSCAN and find the normals of the planes using RANSAC
        @param pcd: open3d.geometry.PointCloud
        @return List[SegmentResult]: list of planes
        '''
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        camera.translate([0, 0, 0])
        pcd.estimate_normals()
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        clusters_count = labels.max() + 1
        cls._logger.info(f"total clusters: {clusters_count}")
        result = []

        for i in range(clusters_count):
            cluster = pcd.select_by_index(np.where(labels == i)[0])
            plane_model, inliers = cluster.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
            inlier_cloud = cluster.select_by_index(inliers)
            # filter out small clusters
            if len(inliers) < 3000:
                continue
         
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(inlier_cloud.points)
            center = bbox.get_center()
            dimensions = (bbox.extent[0], bbox.extent[1])

            density = len(pcd.points) / bbox.volume()
            # filter out low density clusters
            if density < 10e6:
                continue

            cls._logger.info(f"cluster {i} density: {density}, w: {dimensions[0]}, h: {dimensions[1]}")

            R = cls.find_rotation_matrix(plane_model)
            # rotate along y axis by 90 degrees
            R = np.dot(R, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
            plane_tf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            plane_tf.compute_vertex_normals()
            plane_tf.rotate(R)
            plane_tf.translate(center)
            
            inlier_cloud.paint_uniform_color((0.5, 0, 0.5))
            bbox.color = np.random.rand(3)
            tf_matrix = np.eye(4)
            tf_matrix[:3, :3] = R
            tf_matrix[:3, 3] = center
            result.append(SegmentResult(plane_cloud=inlier_cloud, 
                                        plane_tf=plane_tf, 
                                        bbox=bbox, 
                                        dims=dimensions, 
                                        tf_matrix=tf_matrix))
        return result
