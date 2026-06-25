"""Synthetic sensor model: turns truth-rate logs into noisy measurements.

Sensors available to the estimator:
    P1, P2 : pressure transducers (noise + bias + drift + bandwidth + delay + ADC)
    Q1, Q2 : flow meters         (noise + bias + drift + K-factor nonlinearity +
                                  bandwidth + delay + saturation)  <- bias kills
                                                                      naive integration
    T      : oil temperature      (noise + per-run bias)
NO position or velocity sensor is provided -> the scheme is sensorless.
"""
from __future__ import annotations

import numpy as np

from .params import Config


def _quantize(x, full_scale, bits):
    q = full_scale / (2 ** bits)
    return np.round(x / q) * q


def _lowpass(x, fc, dt):
    """Causal first-order low-pass (transducer bandwidth)."""
    if fc is None or fc <= 0:
        return x
    a = np.exp(-2.0 * np.pi * fc * dt)
    y = np.empty_like(x)
    acc = x[0]
    for i in range(x.size):
        acc = a * acc + (1.0 - a) * x[i]
        y[i] = acc
    return y


def _delay(x, n_shift):
    """Pure transport delay by n_shift samples (hold initial value)."""
    if n_shift <= 0:
        return x
    y = np.empty_like(x)
    y[:n_shift] = x[0]
    y[n_shift:] = x[:-n_shift]
    return y


def _drift(n, sigma_per_s, dt, rng):
    """Slow bias random walk (sensor zero drift)."""
    if sigma_per_s <= 0:
        return np.zeros(n)
    steps = rng.normal(0.0, sigma_per_s * np.sqrt(dt), n)
    return np.cumsum(steps)


def make_measurements(truth: dict, cfg: Config, rng: np.random.Generator) -> dict:
    s = cfg.sens
    n = truth["t"].size
    dt = 1.0 / s.sample_rate

    # ----- per-run constant biases (drawn once per run) -----
    p_bias = rng.normal(0.0, s.p_bias_frac * s.p_fullscale, size=2)
    q_bias = rng.normal(0.0, s.q_bias_frac * s.q_fullscale, size=2)
    T_bias = rng.normal(0.0, s.T_bias)
    n_delay = int(round(s.delay_ms * 1e-3 * s.sample_rate))

    # ----- pressure: bandwidth -> noise -> bias + drift -> delay -> ADC -----
    p_sigma = s.p_noise_frac * s.p_fullscale
    p_drift_sig = s.bias_drift_frac * s.p_fullscale

    def make_p(P, bias):
        y = _lowpass(P, s.p_bandwidth, dt)
        y = y + bias + _drift(n, p_drift_sig, dt, rng) + rng.normal(0.0, p_sigma, n)
        y = _delay(y, n_delay)
        return _quantize(y, s.p_fullscale, s.p_adc_bits)

    P1m = make_p(truth["P1"], p_bias[0])
    P2m = make_p(truth["P2"], p_bias[1])

    # ----- flow: K-factor nonlinearity -> bandwidth -> noise -> bias+drift
    #             -> delay -> saturation -----
    q_floor = s.q_noise_floor_frac * s.q_fullscale
    q_drift_sig = s.bias_drift_frac * s.q_fullscale

    def make_q(Q, bias):
        # K-factor nonlinearity: gain error grows with |reading|/FS
        y = Q * (1.0 + s.q_kfactor_nl * (Q / s.q_fullscale))
        y = _lowpass(y, s.q_bandwidth, dt)
        sig = np.hypot(s.q_noise_frac * np.abs(Q), q_floor)
        y = y + bias + _drift(n, q_drift_sig, dt, rng) + rng.normal(0.0, 1.0, n) * sig
        y = _delay(y, n_delay)
        if s.clip_to_fullscale:
            y = np.clip(y, -s.q_fullscale, s.q_fullscale)
        return y

    Q1m = make_q(truth["Q1"], q_bias[0])
    Q2m = make_q(truth["Q2"], q_bias[1])

    # ----- temperature measurement (slow channel; no dynamics) -----
    Tm = truth["T"] + T_bias + rng.normal(0.0, s.T_noise, n)

    return {
        "t": truth["t"],
        "P1": P1m, "P2": P2m,
        "Q1": Q1m, "Q2": Q2m,
        "T": Tm,
        "_true_biases": {"p": p_bias, "q": q_bias, "T": T_bias},
    }
