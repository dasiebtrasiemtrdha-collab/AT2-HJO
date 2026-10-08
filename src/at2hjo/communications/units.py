"""Explicit decimal SI conversions at configuration and data boundaries."""
from __future__ import annotations

import numpy as np


def _finite(value, name="value"):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _result(a):
    return float(a) if np.ndim(a) == 0 else a


def db_to_linear(value):
    """Power/antenna gain dB to dimensionless linear gain (10, not 20)."""
    with np.errstate(over="ignore", under="ignore"):
        a = np.exp(_finite(value) * np.log(10.0) / 10.0)
    if not np.isfinite(a).all():
        raise ValueError("dB conversion overflow")
    return _result(a)


def linear_to_db(value):
    a = _finite(value)
    if np.any(a < 0):
        raise ValueError("linear power gain must be nonnegative")
    with np.errstate(divide="ignore"):
        return _result(10.0 * np.log10(a))


def dbm_to_w(value):
    return _result(np.asarray(db_to_linear(value)) * 1e-3)


def w_to_dbm(value):
    return _result(np.asarray(linear_to_db(value)) + 30.0)


def mhz_to_hz(value): return _result(_finite(value) * 1e6)
def hz_to_mhz(value): return _result(_finite(value) / 1e6)
def ghz_to_hz(value): return _result(_finite(value) * 1e9)
def hz_to_ghz(value): return _result(_finite(value) / 1e9)
def mbit_to_bit(value): return _result(_finite(value) * 1e6)
def bit_to_mbit(value): return _result(_finite(value) / 1e6)
def gb_to_bit(value): return _result(_finite(value) * 8e9)
def bit_to_gb(value): return _result(_finite(value) / 8e9)
def km_to_m(value): return _result(_finite(value) * 1e3)
def m_to_km(value): return _result(_finite(value) / 1e3)
def degree_to_rad(value): return _result(np.deg2rad(_finite(value)))
def rad_to_degree(value): return _result(np.rad2deg(_finite(value)))
