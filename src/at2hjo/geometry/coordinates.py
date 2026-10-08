"""Metre-valued local ENU and spherical-Earth coordinates.

The local simulation square uses east=x, north=y and up=z. Sea surface
is z=0; submerged nodes have negative z. ECI/ECEF vectors are Earth-centred
and must be explicitly transformed before mixing with local positions.
"""
from __future__ import annotations

import numpy as np

NODE_TYPES = frozenset(("BN", "LEO", "AUV", "USN"))


def validate_node_typing(node_ids, node_types) -> None:
    """Validate type labels and typed IDs without integer-range inference."""
    ids, types = np.asarray(node_ids), np.asarray(node_types)
    if ids.ndim != 1 or types.shape != ids.shape:
        raise ValueError("node IDs/types must be matching one-dimensional arrays")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("node IDs must be unique")
    for node_id, node_type in zip(ids, types):
        if str(node_type) not in NODE_TYPES or not str(node_id).startswith(f"{node_type}:"):
            raise ValueError(f"Node ID/type mismatch: {node_id}, {node_type}")
        suffix = str(node_id).split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError("typed node ID requires a nonnegative integer suffix")


def typed_node_indices(scenario: dict, node_type: str) -> np.ndarray:
    validate_node_typing(scenario["node_id"], scenario["node_type"])
    if node_type not in NODE_TYPES:
        raise ValueError(f"Unknown node type: {node_type}")
    return np.flatnonzero(np.asarray(scenario["node_type"]) == node_type)


def pairwise_distance(points_a, points_b) -> np.ndarray:
    """Return Euclidean distances [..., A, B] from [..., A/B, 3] points."""
    a, b = np.asarray(points_a, dtype=np.float64), np.asarray(points_b, dtype=np.float64)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError("pairwise points must have shape [..., nodes, 3]")
    return np.linalg.norm(a[..., :, None, :] - b[..., None, :, :], axis=-1)


def geometric_coverage(underwater, buoys, max_distance_m: float = 150.) -> dict:
    """Geometry only: inclusive distance threshold, with no SNR/rate inference."""
    if not np.isfinite(max_distance_m) or max_distance_m < 0:
        raise ValueError("maximum distance must be finite and nonnegative")
    distance = pairwise_distance(underwater, buoys)
    mask = distance <= max_distance_m
    count = mask.sum(axis=-1, dtype=np.int64)
    return {"distance_m": distance, "in_range": mask, "coverage_count": count,
            "uncovered": count == 0}


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0., earth_radius_m=6371000.) -> np.ndarray:
    """Spherical geodetic coordinates, broadcasting latitude/longitude/altitude."""
    lat, lon, alt = np.broadcast_arrays(np.deg2rad(lat_deg), np.deg2rad(lon_deg), alt_m)
    r = np.asarray(earth_radius_m, dtype=np.float64) + alt
    return np.stack((r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon),
                     r * np.sin(lat)), axis=-1)


def enu_rotation(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows map an ECEF column displacement to local east/north/up."""
    lat, lon = np.deg2rad([lat_deg, lon_deg])
    sl, cl, sp, cp = np.sin(lon), np.cos(lon), np.sin(lat), np.cos(lat)
    return np.asarray([[-sl, cl, 0.], [-sp * cl, -sp * sl, cp],
                       [cp * cl, cp * sl, sp]], dtype=np.float64)


def ecef_to_enu(points_ecef, origin_ecef, lat_deg: float, lon_deg: float) -> np.ndarray:
    return (np.asarray(points_ecef, dtype=np.float64) - np.asarray(origin_ecef)) @ enu_rotation(lat_deg, lon_deg).T


def enu_to_ecef(points_enu, origin_ecef, lat_deg: float, lon_deg: float) -> np.ndarray:
    return np.asarray(points_enu, dtype=np.float64) @ enu_rotation(lat_deg, lon_deg) + np.asarray(origin_ecef)


def _earth_rotate(points, angle):
    p = np.asarray(points, dtype=np.float64)
    if p.shape[-1] != 3:
        raise ValueError("Earth-centred points require a final xyz axis")
    angle = np.asarray(angle, dtype=np.float64)
    while angle.ndim < p.ndim - 1:
        angle = angle[..., None]
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c * p[..., 0] - s * p[..., 1],
                     s * p[..., 0] + c * p[..., 1], p[..., 2]), axis=-1)


def eci_to_ecef(points_eci, time_s, omega=7.292115e-5, angle0=0.) -> np.ndarray:
    """Rotate inertial vectors by minus the declared Earth rotation angle."""
    return _earth_rotate(points_eci, -(np.asarray(time_s) * omega + angle0))


def ecef_to_eci(points_ecef, time_s, omega=7.292115e-5, angle0=0.) -> np.ndarray:
    return _earth_rotate(points_ecef, np.asarray(time_s) * omega + angle0)

