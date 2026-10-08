"""Analytic circular Walker geometry in a spherical, rotating Earth frame.

The declared epoch and Earth angle are synthetic reconstruction parameters.
Propagation uses elapsed SI seconds; no astronomical epoch fitting or channel
budget is implied. Satellites are propagated independently of learner RNGs.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .coordinates import ecef_to_enu, eci_to_ecef, geodetic_to_ecef


def _points(value, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite [..., node, 3] coordinates")
    return points


def circular_orbit_period(altitude_m: float, earth_radius_m: float = 6371000.0,
                          earth_mu_m3_s2: float = 398600441800000.0) -> float:
    """Two-body circular period in seconds, with SI inputs."""
    if (not np.isfinite([altitude_m, earth_radius_m, earth_mu_m3_s2]).all()
            or altitude_m <= 0 or earth_radius_m <= 0 or earth_mu_m3_s2 <= 0):
        raise ValueError("positive altitude, Earth radius and gravitational parameter required")
    return float(2 * np.pi * np.sqrt((earth_radius_m + altitude_m) ** 3 / earth_mu_m3_s2))


def orbital_ephemeris(config_orbit: Mapping, times_s,
                      variant: str = "regional_overhead") -> dict:
    """Propagate the frozen circular Walker realization at arbitrary times.

    Satellite 0 is directly above the origin on the ascending orbital branch
    at reset in ``regional_overhead``. ``walker_stress`` offsets the anomaly
    by half one satellite spacing; a pi offset in the six-satellite ring would
    merely permute IDs. Both variants subsequently propagate real geometry.
    Array axes are [time, satellite, coordinate], in metres.
    """
    if variant not in {"regional_overhead", "walker_stress", "regional_walker_candidates"}:
        raise ValueError(f"unknown orbital geometry variant: {variant}")
    if variant == "regional_walker_candidates":
        # Opt-in reconstruction; the accepted six-satellite v1 realization
        # remains unchanged. Selection happens once over declared geometry,
        # never by replacing candidates during an episode.
        from at2hjo.data.provisioning import walker_candidate_ephemeris
        return walker_candidate_ephemeris(config_orbit, times_s)
    times = np.atleast_1d(np.asarray(times_s, dtype=np.float64))
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times_s must be a finite one-dimensional sequence")
    radius = float(config_orbit["earth_radius_m"])
    altitude = float(config_orbit["altitude_m"])
    mu = float(config_orbit["earth_mu_m3_s2"])
    period = circular_orbit_period(altitude, radius, mu)
    n = 2 * np.pi / period
    planes = int(config_orbit["planes"])
    per_plane = int(config_orbit["satellites_per_plane"])
    if planes <= 0 or per_plane <= 0:
        raise ValueError("positive Walker plane and satellite counts required")
    inclination = np.deg2rad(float(config_orbit["inclination_deg"]))
    latitude = np.deg2rad(float(config_orbit["origin_lat_deg"]))
    longitude = np.deg2rad(float(config_orbit["origin_lon_deg"]))
    if not 0 <= inclination <= np.pi or not -np.pi / 2 <= latitude <= np.pi / 2:
        raise ValueError("inclination or reference latitude outside physical range")
    sin_i = np.sin(inclination)
    if abs(sin_i) < 1e-14:
        if abs(latitude) > 1e-14:
            raise ValueError("equatorial orbit cannot pass over the selected latitude")
        anomaly_origin = 0.0
    else:
        ratio = np.sin(latitude) / sin_i
        if abs(ratio) > 1 + 1e-12:
            raise ValueError("orbit inclination cannot pass over the selected latitude")
        anomaly_origin = float(np.arcsin(np.clip(ratio, -1.0, 1.0)))
    angle0 = float(config_orbit.get("earth_angle_at_epoch_rad", 0.0))
    orbital_longitude = np.arctan2(np.cos(inclination) * np.sin(anomaly_origin),
                                  np.cos(anomaly_origin))
    raan_origin = longitude + angle0 - orbital_longitude
    plane_index = np.repeat(np.arange(planes), per_plane)
    in_plane_index = np.tile(np.arange(per_plane), planes)
    raan = raan_origin + 2 * np.pi * plane_index / planes
    phase = float(config_orbit.get("walker_phase", 0))
    anomaly0 = (anomaly_origin + 2 * np.pi * in_plane_index / per_plane
                + 2 * np.pi * phase * plane_index / (planes * per_plane))
    if variant == "walker_stress":
        anomaly0 = anomaly0 + np.pi / per_plane
    anomaly = np.remainder(anomaly0[None, :] + n * times[:, None], 2 * np.pi)
    cos_u, sin_u = np.cos(anomaly), np.sin(anomaly)
    cos_o, sin_o = np.cos(raan)[None, :], np.sin(raan)[None, :]
    r = radius + altitude
    eci = np.stack((r * (cos_o * cos_u - sin_o * sin_u * np.cos(inclination)),
                    r * (sin_o * cos_u + cos_o * sin_u * np.cos(inclination)),
                    r * sin_u * np.sin(inclination)), axis=-1)
    omega = float(config_orbit["earth_rotation_rad_s"])
    ecef = eci_to_ecef(eci, times, omega=omega, angle0=angle0)
    origin = geodetic_to_ecef(np.rad2deg(latitude), np.rad2deg(longitude),
                              earth_radius_m=radius)
    local_enu = ecef_to_enu(ecef, origin, np.rad2deg(latitude), np.rad2deg(longitude))
    return {"times_s": times, "eci_m": eci, "ecef_m": ecef,
            "local_enu_m": local_enu, "period_s": period,
            "mean_motion_rad_s": float(n), "altitude_m": altitude,
            "orbital_radius_m": r, "inclination_deg": float(np.rad2deg(inclination)),
            "raan_rad": raan,
            "initial_anomaly_rad": anomaly0, "plane_index": plane_index,
            "in_plane_index": in_plane_index, "variant": variant,
            "epoch_utc": str(config_orbit["epoch_utc"]),
            "origin_ecef_m": origin}


def segment_min_radius(first, second) -> np.ndarray:
    """Minimum distance of a closed line segment from the Earth centre."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    delta = b - a
    squared_length = np.sum(delta * delta, axis=-1)
    t = np.divide(-np.sum(a * delta, axis=-1), squared_length,
                  out=np.zeros_like(squared_length), where=squared_length > 0)
    closest = a + np.clip(t, 0.0, 1.0)[..., None] * delta
    return np.linalg.norm(closest, axis=-1)


def ground_visibility(sat_ecef, observer_ecef, min_elevation_deg: float = 10.0,
                      earth_radius: float = 6371000.0) -> dict[str, np.ndarray]:
    """Ranges and local sky geometry for every observer-satellite pair.

    Output axes [..., observer, satellite]. Earth occultation tests the finite
    chord against the spherical Earth interior. The exact horizon tangent is
    unocculted; the separate elevation threshold still applies.
    """
    if earth_radius <= 0 or not -90 <= min_elevation_deg <= 90:
        raise ValueError("invalid Earth radius or elevation threshold")
    satellites = _points(sat_ecef, "sat_ecef")
    observers = _points(observer_ecef, "observer_ecef")
    observer_norm = np.linalg.norm(observers, axis=-1)
    if np.any(observer_norm < earth_radius - 1e-6):
        raise ValueError("ground observer lies inside spherical Earth")
    delta = satellites[..., None, :, :] - observers[..., :, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    if np.any(distance == 0):
        raise ValueError("satellite and observer cannot coincide")
    up = observers / observer_norm[..., None]
    lon = np.arctan2(observers[..., 1], observers[..., 0])
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.cross(up, east)
    local = np.stack((np.sum(delta * east[..., :, None, :], axis=-1),
                      np.sum(delta * north[..., :, None, :], axis=-1),
                      np.sum(delta * up[..., :, None, :], axis=-1)), axis=-1)
    elevation = np.arctan2(local[..., 2], np.linalg.norm(local[..., :2], axis=-1))
    azimuth = np.mod(np.arctan2(local[..., 0], local[..., 1]), 2 * np.pi)
    minimum = segment_min_radius(observers[..., :, None, :], satellites[..., None, :, :])
    # Sub-micrometre tolerance suppresses round-off at a tangent or surface.
    occulted = minimum < earth_radius - 1e-6
    visible = (~occulted) & (elevation >= np.deg2rad(min_elevation_deg) - 1e-12)
    return {"distance_m": distance, "elevation_rad": elevation,
            "azimuth_rad": azimuth, "local_enu_m": local,
            "visible": visible, "occulted": occulted,
            "earth_los": ~occulted, "chord_min_radius_m": minimum}


def isl_geometry(sat_ecef, orbitconfig: Mapping) -> dict[str, np.ndarray]:
    """Finite-chord Earth clearance and configured ISL candidate topology.

    ``candidate_neighbors`` is a topological choice; ``available_neighbors``
    additionally requires unobstructed geometry and the configured distance
    bound. These masks contain no channel gain, bandwidth or rate.
    """
    satellites = _points(sat_ecef, "sat_ecef")
    count = satellites.shape[-2]
    planes = int(orbitconfig["planes"])
    per_plane = int(orbitconfig["satellites_per_plane"])
    if count != planes * per_plane:
        raise ValueError("satellite count differs from Walker plane configuration")
    first, second = satellites[..., :, None, :], satellites[..., None, :, :]
    distance = np.linalg.norm(first - second, axis=-1)
    chord_radius = segment_min_radius(first, second)
    radius = float(orbitconfig["earth_radius_m"])
    clearance = float(orbitconfig.get("isl_earth_clearance_m", 0.0))
    max_distance = float(orbitconfig["isl_max_distance_m"])
    if radius <= 0 or clearance < 0 or max_distance <= 0:
        raise ValueError("invalid ISL geometric limits")
    not_self = ~np.eye(count, dtype=np.bool_)
    earth_los = (chord_radius >= radius + clearance - 1e-6) & not_self
    rule = orbitconfig.get("isl_neighbors", "adjacent_ring")
    candidate = np.zeros((count, count), dtype=np.bool_)
    if rule == "adjacent_ring":
        for plane in range(planes):
            for local in range(per_plane):
                node = plane * per_plane + local
                candidate[node, plane * per_plane + (local - 1) % per_plane] = True
                candidate[node, plane * per_plane + (local + 1) % per_plane] = True
        candidate &= not_self
    elif rule == "same_adjacent_planes":
        for plane in range(planes):
            for local in range(per_plane):
                node = plane * per_plane + local
                for dp, ds in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    candidate[node, ((plane + dp) % planes) * per_plane
                              + (local + ds) % per_plane] = True
        candidate &= not_self
    elif rule == "nearest_visible":
        limit = int(orbitconfig.get("isl_max_neighbors", 2))
        if limit < 0:
            raise ValueError("isl_max_neighbors must be nonnegative")
        candidates = np.where(earth_los & (distance <= max_distance), distance, np.inf)
        order = np.argsort(candidates, axis=-1, kind="stable")
        candidate = np.zeros(distance.shape, dtype=np.bool_)
        for rank in range(min(limit, max(count - 1, 0))):
            index = order[..., rank:rank + 1]
            finite = np.isfinite(np.take_along_axis(candidates, index, axis=-1))
            np.put_along_axis(candidate, index, finite, axis=-1)
        # Mutual nearest candidates preserve reciprocity and the degree limit.
        candidate &= np.swapaxes(candidate, -1, -2)
    else:
        raise ValueError(f"unsupported ISL neighbor rule: {rule}")
    candidate = np.broadcast_to(candidate, distance.shape).copy()
    available = candidate & earth_los & (distance <= max_distance)
    return {"distance_m": distance, "chord_min_radius_m": chord_radius,
            "earth_los": earth_los, "candidate_neighbors": candidate,
            "available_neighbors": available}


def visibility_windows(visible, times_s, slot_duration_s: float) -> list[dict]:
    """Sampled half-open visibility windows [start_s,end_s) per observer/sat.

    Durations count observed slots; no sub-slot crossing interpolation or
    unobserved continuation past the supplied interval is inferred.
    """
    mask = np.asarray(visible, dtype=np.bool_)
    times = np.asarray(times_s, dtype=np.float64)
    if mask.ndim != 3 or times.shape != (mask.shape[0],) or slot_duration_s <= 0:
        raise ValueError("expected [time,observer,satellite] visibility and matching times")
    if not np.isfinite(times).all() or (times.size > 1 and
            not np.allclose(np.diff(times), slot_duration_s, rtol=0, atol=1e-8)):
        raise ValueError("visibility windows require a finite regular slot grid")
    windows = []
    for observer in range(mask.shape[1]):
        for satellite in range(mask.shape[2]):
            padded = np.concatenate(([False], mask[:, observer, satellite], [False]))
            transitions = np.diff(padded.astype(np.int8))
            for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
                windows.append({"observer_index": observer, "satellite_index": satellite,
                                "start_st": int(start), "end_st_exclusive": int(end),
                                "start_s": float(times[start]),
                                "end_s": float(times[end - 1] + slot_duration_s),
                                "duration_s": float((end - start) * slot_duration_s),
                                "left_censored": bool(start == 0),
                                "right_censored": bool(end == len(times))})
    return windows
