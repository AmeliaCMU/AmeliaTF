import numpy as np
import torch

from geographiclib.geodesic import Geodesic
from math import sin, cos, radians
from torch import tensor
from typing import Tuple

def wrap_angle(angle):
    return np.radians(((angle % 360) + 540) % 360 - 180)

def transform_points_2d(points: np.array, ref_point: np.array, theta: float) -> np.array:
    """" Transforms a set of inputs points forllowing: P = (P - t) @ R. 

    Inputs
    ------
        points[np.array]: input points to be transformed.
        ref_point[np.array]: point used to translate the input points. 
        theta[float]: rotation angle. 

    Outputs
    -------
        tf_points[np.array]: transformed points. 
    """
    tf_points = points.copy()
    R = np.array(
        [[cos(theta), -sin(theta)], 
         [sin(theta),  cos(theta)]])
    tf_points[:, 0:2] = np.matmul(points[:, 0:2] - ref_point.reshape(1, 2), R)
    tf_points[:, 2:4] = np.matmul(points[:, 2:4] - ref_point.reshape(1, 2), R)
    return tf_points

def inv_transform(traj_rel: np.array, start_abs: np.array, theta: float) -> np.array:
    """" Transforms a set of inputs points forllowing: P = (P - t) @ R. 

    Inputs
    ------
        traj_rel[np.array]: Relative trajectory
        start_abs[np.array]: point used to translate the input points. 
        theta[float]: rotation angle. 

    Outputs
    -------
        tf_points[np.array]: transformed points. 
    """
    heading = radians(theta)
    # heading = -radians(theta)
    R = np.array(
        [[cos(heading), -sin(heading)], 
         [sin(heading),  cos(heading)]])
    rot_coords = traj_rel @ R.T 
    return rot_coords + start_abs

def xy_to_ll(
    traj_rel: tensor, start_abs_xy: tensor, start_heading: tensor, reference: Tuple, geodesic: Geodesic
) -> np.array:
    """
    
    Inputs
    ------
        mu (tensor): tensor containing model's prediction in relative XY
        hist_abs (tensor): tensor containg past trajectory in absolute XY
        reference (Tuple): tuple containing the reference lat/lon points
        geodesic (Geodesic): geode for computing lat/lon

    Returns:
        tensor: tensor containing mu's values in lat/lon.
    """
    N, _ , _ = traj_rel.shape
    traj_ll = torch.zeros_like(traj_rel)
    traj_xy_abs = inv_transform(traj_rel.cpu().numpy(), start_abs_xy, start_heading)

    for n in range(N):
        x, y = traj_xy_abs[n, :, 0], traj_xy_abs[n, :, 1]
        rang = np.sqrt(x ** 2 + y ** 2)
        bearing = np.degrees(np.arctan2(y, x))
        # lat, lon
        lat, lon = direct_wrapper(geodesic, bearing, rang, reference[0], reference[1], reference[2])
        traj_ll[n , :, 1] = torch.tensor(lon) 
        traj_ll[n , :, 0] = torch.tensor(lat)
    return traj_ll

def direct_wrapper(geodesic, b, r, ref_lat, ref_lon, r_scale):
    """ Computes lat/lon from range (r) and bearing (b).
    Inputs:
    -------
        mu (tensor): tensor containing model's prediction in relative XY
        hist_abs (tensor): tensor containg past trajectory in absolute XY
        reference (Tuple): tuple containing the reference lat/lon points
        geodesic (Geodesic): geode for computing lat/lon

    Returns:
        tensor: tensor containing mu's values in lat/lon.
    """
    lat_array = []
    lon_array = []
    for i in range(r.shape[0]):
        g = geodesic.Direct(ref_lat, ref_lon, b[i], r[i] * r_scale)
        lat_array.append(g['lat2'])
        lon_array.append(g['lon2'])
    return lat_array, lon_array