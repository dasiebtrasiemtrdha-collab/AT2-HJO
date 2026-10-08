"""Non-mutating conversion of the frozen communication configuration to SI."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .units import db_to_linear, dbm_to_w, degree_to_rad, ghz_to_hz, mhz_to_hz, km_to_m


def _convert_alias(group, source, target, convert):
    if source in group:
        if target in group:
            raise ValueError(f"ambiguous units: both {source} and {target}")
        group[target] = convert(group.pop(source))


def load_communication_config(config) -> dict:
    """Return channels only, with angles in rad and every power gain linear.

    A full resolved scientific dict, ResolvedConfig, or a channels dict is
    accepted. Aliases are converted exactly once; input mappings are untouched.
    The frozen scientific values are never replaced with fitted defaults.
    """
    raw = config.config if hasattr(config, "config") else config
    channels = deepcopy(raw.get("channels", raw))
    _convert_alias(channels, "noise_psd_dbm_hz", "noise_psd_w_hz", dbm_to_w)
    for name in ("optical", "bn", "rf", "isl"):
        g = channels[name]
        for source, target, fn in (
            ("bandwidth_mhz", "bandwidth_hz", mhz_to_hz),
            ("carrier_ghz", "carrier_hz", ghz_to_hz),
            ("tx_power_dbm", "tx_power_w", dbm_to_w),
            ("max_distance_km", "max_distance_m", km_to_m),
            ("divergence_deg", "divergence_rad", degree_to_rad),
            ("tx_gain_dbi", "tx_gain_linear", db_to_linear),
            ("rx_gain_dbi", "rx_gain_linear", db_to_linear),
            ("extra_loss_db", "extra_loss_linear", db_to_linear),
            ("rician_k_db", "rician_k_linear", db_to_linear),
        ):
            _convert_alias(g, source, target, fn)
        for key in ("tx_power_w", "bandwidth_hz"):
            if not np.isfinite(g[key]) or g[key] < 0:
                raise ValueError(f"{name}.{key} must be finite and nonnegative")
        if name == "optical":
            if (not np.isfinite([g["aperture_m2"], g["divergence_rad"], g["cw_m_inv"], g["max_distance_m"]]).all()
                or g["aperture_m2"] <= 0 or not 0 < g["divergence_rad"] <= np.pi
                or g["cw_m_inv"] < 0 or g["max_distance_m"] <= 0):
                raise ValueError("invalid optical SI parameters")
        else:
            required = [g["carrier_hz"], g["tx_gain_linear"], g["rx_gain_linear"], g["extra_loss_linear"]]
            if not np.isfinite(required).all() or np.any(np.asarray(required) <= 0):
                raise ValueError(f"invalid {name} RF SI parameters")
            if "rician_k_linear" in g and (not np.isfinite(g["rician_k_linear"]) or g["rician_k_linear"] < 0):
                raise ValueError("Rician K must be finite and nonnegative")
            if "shadow_std_db" in g and (not np.isfinite(g["shadow_std_db"]) or g["shadow_std_db"] < 0):
                raise ValueError("shadow standard deviation must be finite and nonnegative")
    if (not np.isfinite([channels["noise_psd_w_hz"], channels["speed_of_light_m_s"], channels["snr_threshold_linear"]]).all()
        or channels["noise_psd_w_hz"] <= 0 or channels["speed_of_light_m_s"] <= 0
        or channels["snr_threshold_linear"] < 0):
        raise ValueError("invalid noise, light speed or SNR threshold")
    if channels["probe_bandwidth"] != "full_pool_before_action" or channels["probe_mask_use"] != "conservative_fixed_within_slot":
        raise NotImplementedError("only the frozen full-pool conservative probe protocol is implemented")
    return channels
