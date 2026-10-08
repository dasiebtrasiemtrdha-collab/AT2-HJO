"""Four communication families: paper Eq(1), (3)--(11), in SI units.

BN/RF/ISL gains are the frozen Friis/Rician loss approximations from Stage02,
not a replacement for an undisclosed full wave/duct model. All stochastic
values are supplied CPU primitives; no function draws from any RNG.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np

from .configuration import load_communication_config


def _scalar(a):
    return float(a) if np.ndim(a) == 0 else a


def optical_gain(distance_m, aperture_m2, divergence_rad, extinction_m_inv):
    """Eq(3), returning NaN for singular/invalid geometry and 0 on underflow.

    No undocumented near-field distance floor or gain cap modifies the paper
    equation. A nonpositive distance is rejected by LinkBudget as numerical_invalid.
    """
    d, a, theta, c = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in
                                          (distance_m, aperture_m2, divergence_rad, extinction_m_inv)])
    valid = (np.isfinite(d) & np.isfinite(a) & np.isfinite(theta) & np.isfinite(c)
             & (d > 0) & (a > 0) & (theta > 0) & (theta <= np.pi) & (c >= 0))
    out = np.full(d.shape, np.nan, dtype=np.float64)
    # 2 sin²(theta/2) avoids loss of precision in 1-cos(theta).
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        log_gain = (np.log(2 * a) - math.log(math.pi) - 2 * np.log(d)
                    - np.log(2 * np.sin(theta / 2) ** 2) - c * d)
        gain = np.exp(log_gain)
    out[valid & np.isfinite(gain)] = gain[valid & np.isfinite(gain)]
    return _scalar(out)


def friis_gain(distance_m, carrier_hz, tx_gain_linear=1., rx_gain_linear=1.,
               extra_loss_linear=1., speed_of_light_m_s=299792458.):
    """GT*GR*(c/(4πfd))²/L, with dimensionless power gain and SI inputs."""
    d, f, gt, gr, loss, c = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in
        (distance_m, carrier_hz, tx_gain_linear, rx_gain_linear, extra_loss_linear, speed_of_light_m_s)])
    valid = np.isfinite(d) & np.isfinite(f) & np.isfinite(gt) & np.isfinite(gr) & np.isfinite(loss) & np.isfinite(c)
    valid &= (d > 0) & (f > 0) & (gt > 0) & (gr > 0) & (loss > 0) & (c > 0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        gain = np.exp(np.log(gt) + np.log(gr) - np.log(loss)
                      + 2 * (np.log(c) - math.log(4 * math.pi) - np.log(f) - np.log(d)))
    return _scalar(np.where(valid & np.isfinite(gain), gain, np.nan))


def rician_power(gaussian_real_imag, k_linear):
    """Unit-mean Rician power from two existing independent N(0,1) values."""
    z = np.asarray(gaussian_real_imag, dtype=np.float64)
    k = np.asarray(k_linear, dtype=np.float64)
    if z.shape[-1:] != (2,):
        raise ValueError("Rician primitive must have final real/imag axis of length 2")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        real = np.sqrt(k / (k + 1)) + z[..., 0] / np.sqrt(2 * (k + 1))
        imaginary = z[..., 1] / np.sqrt(2 * (k + 1))
        power = real * real + imaginary * imaginary
    valid = np.isfinite(z).all(axis=-1) & np.isfinite(k) & (k >= 0) & np.isfinite(power)
    return _scalar(np.where(valid, power, np.nan))


def shadow_power(standard_normal, std_db):
    """Power attenuation 10^(-sigma*z/10); positive z is extra loss."""
    z, sigma = np.broadcast_arrays(np.asarray(standard_normal, dtype=np.float64), np.asarray(std_db, dtype=np.float64))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        power = np.exp(-sigma * z * math.log(10) / 10)
    return _scalar(np.where(np.isfinite(z) & np.isfinite(sigma) & (sigma >= 0) & np.isfinite(power), power, np.nan))


def snr(tx_power_w, channel_gain, noise_psd_w_hz, bandwidth_hz):
    """Eq(1); zero bandwidth gives defined zero SNR (no transmission)."""
    p, g, n, b = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64) for x in
                                      (tx_power_w, channel_gain, noise_psd_w_hz, bandwidth_hz)])
    valid = np.isfinite(p) & np.isfinite(g) & np.isfinite(n) & np.isfinite(b) & (p >= 0) & (g >= 0) & (n > 0) & (b >= 0)
    result = np.zeros(b.shape, dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        np.divide(p * g, n * b, out=result, where=valid & (b > 0))
    return _scalar(np.where(valid & np.isfinite(result), result, np.nan))


def shannon_rate(bandwidth_hz, snr_linear, available=True):
    """Eq(10); a closed mask or zero bandwidth produces exactly zero rate."""
    b, gamma, mask = np.broadcast_arrays(np.asarray(bandwidth_hz, dtype=np.float64),
                                         np.asarray(snr_linear, dtype=np.float64), np.asarray(available, dtype=bool))
    active = mask & (b != 0)
    valid = np.isfinite(b) & (b >= 0) & np.isfinite(gamma) & (gamma >= 0)
    with np.errstate(over="ignore", invalid="ignore"):
        rate = b * np.log1p(gamma) / math.log(2)
    result = np.where(active, np.where(valid & np.isfinite(rate), rate, np.nan), 0.)
    # Negative/nonfinite allocated bandwidth is still an invalid input, even masked.
    result = np.where(np.isfinite(b) & (b >= 0), result, np.nan)
    return _scalar(result)


@dataclass(frozen=True)
class LinkBudget:
    link_type: str
    src: str
    dst: str
    distance_m: float
    geometric_available: bool
    external_available: bool
    channel_gain: float
    reference_bandwidth_hz: float
    reference_snr: float
    estimated_rate_bps: float
    tx_power_w: float
    noise_psd_w_hz: float
    snr_threshold_linear: float
    available: bool
    reason_if_unavailable: tuple[str, ...] = ()
    allocated_bandwidth_hz: float | None = None
    actual_snr: float | None = None
    actual_rate_bps: float | None = None

    @property
    def received_power_w(self):
        return self.tx_power_w * self.channel_gain

    @property
    def reference_noise_w(self):
        return self.noise_psd_w_hz * self.reference_bandwidth_hz

    @property
    def path_loss_linear(self):
        return 1. / self.channel_gain if self.channel_gain > 0 else math.inf

    def allocate(self, bandwidth_hz: float) -> "LinkBudget":
        return actual_link_budget(self, bandwidth_hz)


def make_link_budget(link_type, src, dst, distance_m, channel_gain, *, tx_power_w,
                     noise_psd_w_hz, reference_bandwidth_hz, snr_threshold_linear=1.,
                     geometric_available=True, external_available=True,
                     geometry_reason="no_geometric_link", external_reason="external_unavailable"):
    """Build immutable action-independent probe state, retaining every failure gate."""
    reasons = []
    if not geometric_available: reasons.append(geometry_reason)
    if not external_available: reasons.append(external_reason)
    values = np.asarray([distance_m, channel_gain, tx_power_w, noise_psd_w_hz,
                         reference_bandwidth_hz, snr_threshold_linear], dtype=np.float64)
    numeric = (np.isfinite(values).all() and distance_m > 0 and channel_gain >= 0
               and tx_power_w >= 0 and noise_psd_w_hz > 0 and reference_bandwidth_hz >= 0
               and snr_threshold_linear >= 0)
    gamma = float(snr(tx_power_w, channel_gain, noise_psd_w_hz, reference_bandwidth_hz)) if numeric else 0.
    if not numeric or not math.isfinite(gamma):
        reasons.append("numerical_invalid"); gamma = 0.
    elif reference_bandwidth_hz == 0:
        reasons.append("zero_bandwidth")
    elif gamma < snr_threshold_linear:
        reasons.append("below_snr")
    available = not reasons
    rate = float(shannon_rate(reference_bandwidth_hz, gamma, available)) if numeric else 0.
    if not math.isfinite(rate):
        reasons.append("numerical_invalid"); available = False; rate = 0.
    return LinkBudget(str(link_type), str(src), str(dst), float(distance_m), bool(geometric_available),
                      bool(external_available), float(channel_gain) if np.isfinite(channel_gain) and channel_gain >= 0 else 0.,
                      float(reference_bandwidth_hz), gamma, rate, float(tx_power_w), float(noise_psd_w_hz),
                      float(snr_threshold_linear), available, tuple(dict.fromkeys(reasons)))


def actual_link_budget(link: LinkBudget, bandwidth_hz: float) -> LinkBudget:
    """Recompute actual Eq(1)/(10), retaining the conservative pre-action gate.

    Pool overbooking is checked by the routing resource API. Invalid allocations
    fail explicitly here instead of silently producing infinite rates.
    """
    bandwidth = float(bandwidth_hz)
    reasons = list(link.reason_if_unavailable)
    if not math.isfinite(bandwidth) or bandwidth < 0:
        reasons.append("numerical_invalid"); gamma = 0.; rate = 0.
    elif bandwidth == 0:
        reasons.append("zero_bandwidth"); gamma = 0.; rate = 0.
    else:
        gamma = float(snr(link.tx_power_w, link.channel_gain, link.noise_psd_w_hz, bandwidth))
        if not math.isfinite(gamma):
            reasons.append("numerical_invalid"); gamma = 0.
        elif gamma < link.snr_threshold_linear:
            reasons.append("below_snr")
        rate = float(shannon_rate(bandwidth, gamma, link.available and not reasons))
        if not math.isfinite(rate):
            reasons.append("numerical_invalid"); rate = 0.
    return replace(link, available=link.available and not reasons,
                   reason_if_unavailable=tuple(dict.fromkeys(reasons)), allocated_bandwidth_hz=bandwidth,
                   actual_snr=gamma, actual_rate_bps=rate)


def _rf_gain(distance, parameters, channels):
    return friis_gain(distance, parameters["carrier_hz"], parameters["tx_gain_linear"],
                      parameters["rx_gain_linear"], parameters["extra_loss_linear"], channels["speed_of_light_m_s"])


def _groups(scenario):
    from at2hjo.geometry.coordinates import validate_node_typing
    ids, types = np.asarray(scenario["node_id"]), np.asarray(scenario["node_type"])
    validate_node_typing(ids, types)
    return {"underwater": ids[np.isin(types, ["AUV", "USN"])], "bn": ids[types == "BN"], "isl": ids[types == "LEO"]}


def build_link_budgets(scenario, slot, config, case="nominal") -> list[LinkBudget]:
    """All directed candidate pairs, including unavailable diagnostic budgets.

    `case` is a label only: extinction/outages are read from saved slot primitives,
    never regenerated or overridden here. Graph construction keeps available edges.
    """
    c = load_communication_config(config); groups = _groups(scenario); budgets = []
    uw, bn, leo = groups["underwater"], groups["bn"], groups["isl"]
    def add(kind, src, dst, distance, gain, geometry, external, geometry_reason="no_geometric_link", external_reason="external_unavailable"):
        p = c[kind]
        budgets.append(make_link_budget(kind, src, dst, distance, gain,
            tx_power_w=p["tx_power_w"], noise_psd_w_hz=c["noise_psd_w_hz"],
            reference_bandwidth_hz=p["bandwidth_hz"], snr_threshold_linear=c["snr_threshold_linear"],
            geometric_available=geometry, external_available=external,
            geometry_reason=geometry_reason, external_reason=external_reason))
    extinction = float(np.asarray(slot.get("optical_extinction_m_inv", c["optical"]["cw_m_inv"])))
    o = c["optical"]
    for i, source in enumerate(uw):
        for j, dest in enumerate(bn):
            d = float(slot["optical_distance_m"][i, j])
            geom = bool(slot["optical_geometry_valid"][i, j] and slot["optical_geometric_in_range"][i, j] and d <= o["max_distance_m"])
            add("optical", source, dest, d, optical_gain(d, o["aperture_m2"], o["divergence_rad"], extinction), geom, True)
    p = c["bn"]
    for i, source in enumerate(bn):
        for j, dest in enumerate(bn):
            if i == j: continue
            a, b = (min(i, j), max(i, j)) if c["reciprocal_bn_isl"] else (i, j)
            d = float(slot["bn_distance_m"][i, j])
            gain = (_rf_gain(d, p, c) * rician_power(slot["bn_fading_gaussian"][a, b], p["rician_k_linear"])
                    * shadow_power(slot["bn_shadow_standard_normal"][a, b], p["shadow_std_db"]))
            geom = bool(slot["bn_geometry_valid"][i, j] and d <= p["max_distance_m"])
            add("bn", source, dest, d, gain, geom, not bool(slot["bn_wave_blocked"][a, b]), external_reason="wave_blockage")
    p = c["rf"]
    for i, source in enumerate(bn):
        for j, dest in enumerate(leo):
            d = float(slot["rf_distance_m"][i, j]); gain = _rf_gain(d, p, c) * rician_power(slot["rf_fading_gaussian"][i, j], p["rician_k_linear"])
            geom = bool(slot["rf_geometry_valid"][i, j] and slot["rf_visible"][i, j])
            for src, dst in ((source, dest), (dest, source)):
                add("rf", src, dst, d, gain, geom, not bool(slot["rf_outage"][i, j]), "no_visible_satellite", "rf_outage")
    p = c["isl"]
    for i, source in enumerate(leo):
        for j, dest in enumerate(leo):
            if i == j: continue
            a, b = (min(i, j), max(i, j)) if c["reciprocal_bn_isl"] else (i, j)
            d = float(slot["isl_distance_m"][i, j])
            geom = bool(slot["isl_geometry_valid"][i, j] and slot["isl_earth_los"][i, j]
                        and slot["isl_candidate_neighbors"][i, j] and slot["isl_available_neighbors"][i, j])
            add("isl", source, dest, d, _rf_gain(d, p, c), geom, not bool(slot["isl_outage"][a, b]), external_reason="isl_outage")
    return budgets


def ground_station_link_budgets(scenario, slot, config) -> list[LinkBudget]:
    """Deployment-only GS ingress, using declared ground geometry and RF budget.

    Saved `ground_fading_gaussian[S,2]` uses the independent indexed channel_base
    stream. Fixtures lacking this primitive use unit power (no hidden RNG).
    The GS is never an execution node. Its nominal external outage probability is 0.
    """
    from at2hjo.geometry.coordinates import geodetic_to_ecef, enu_to_ecef
    from at2hjo.geometry.orbits import ground_visibility
    raw = config.config if hasattr(config, "config") else config
    channels = load_communication_config(raw); p = channels["rf"]; orbit = raw["orbit"]
    lat, lon, radius = orbit["origin_lat_deg"], orbit["origin_lon_deg"], orbit["earth_radius_m"]
    origin = geodetic_to_ecef(lat, lon, earth_radius_m=radius)
    observer = enu_to_ecef(np.asarray(orbit["ground_station_offset_enu_m"])[None, :], origin, lat, lon)
    geometry = ground_visibility(slot["satellite_ecef_m"], observer, orbit["min_elevation_deg"], radius)
    leo = _groups(scenario)["isl"]; budgets = []
    for j, dest in enumerate(leo):
        distance = float(geometry["distance_m"][0, j])
        fade = rician_power(slot["ground_fading_gaussian"][j], p["rician_k_linear"]) if "ground_fading_gaussian" in slot else 1.
        outage = bool(slot["ground_outage"][j]) if "ground_outage" in slot else False
        budgets.append(make_link_budget("ground", "GS:service_source", dest, distance, _rf_gain(distance, p, channels) * fade,
            tx_power_w=p["tx_power_w"], noise_psd_w_hz=channels["noise_psd_w_hz"],
            reference_bandwidth_hz=p["bandwidth_hz"], snr_threshold_linear=channels["snr_threshold_linear"],
            geometric_available=bool(geometry["visible"][0, j]), external_available=not outage,
            geometry_reason="no_visible_satellite", external_reason="ground_outage"))
    return budgets


# Short descriptive aliases for downstream provider/routing integrations.
ground_station_links = ground_station_link_budgets
