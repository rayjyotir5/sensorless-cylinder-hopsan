"""Sensorless position & force observer.

Position is *not* measured. The observer reconstructs it from the two
physical channels that actually carry the information:

  1. Velocity from flow continuity.  For each chamber, rearranging the
     compressibility equation gives an estimate of piston velocity from the
     measured flow, corrected for flow-meter bias, cross-piston leakage and
     the fluid-compressibility term (V(x)/beta) * dP/dt:

         chamber 1:  A1*v = (Q1 - b1) - (V1/beta)*dP1 - q_leak
         chamber 2:  A2*v = (V2/beta)*dP2 - (Q2 - b2) - q_leak

     The two estimates are blended by flow magnitude (better SNR = more
     weight).  Position is the time integral of the blended velocity.

  2. Flow-meter bias calibration at end-stops.  The slow, integration-
     killing flow-meter bias is observed wherever the piston is seated
     against a hard stop: there the true velocity is zero, so the measured
     flow (minus the known leakage) *is* the bias.  Seating also pins the
     absolute position (homing), bounding integration drift.

Force is obtained directly from the (bias-calibrated) pressure transducers,
F = P1*A1 - P2*A2, which is the standard hydraulic force measurement.

Temperature (measured, +/-1 degC) feeds the beta(T) / viscosity(T) /
leakage(T) models so the corrections track the 0-50 degC envelope.
"""
from __future__ import annotations

import math

from .params import Config


class CylinderObserver:
    def __init__(self, cfg: Config, *, use_bias_calib=True, use_steady_gate=True,
                 use_leak=True, use_compress=True, use_temp=True,
                 single_chamber=False, use_homing=True, use_beta_p=False):
        self.cfg = cfg
        self.A1 = cfg.cyl.area_cap
        self.A2 = cfg.cyl.area_rod
        self.dt = 1.0 / cfg.sens.sample_rate
        # ablation flags (all-on = the full observer)
        self.use_bias_calib = use_bias_calib
        self.use_steady_gate = use_steady_gate
        self.use_leak = use_leak
        self.use_compress = use_compress
        self.use_temp = use_temp
        self.single_chamber = single_chamber
        self.use_homing = use_homing
        self.use_beta_p = use_beta_p   # model entrained-air beta(P) from measured P

        # estimator states
        self.x = 0.0           # position estimate
        self.b1 = 0.0          # flow-meter bias, chamber 1
        self.b2 = 0.0          # flow-meter bias, chamber 2

        # filtered signals
        self._init = False
        self.P1f = cfg.circ.tank_pressure
        self.P2f = cfg.circ.tank_pressure
        self.P1f_prev = self.P1f
        self.P2f_prev = self.P2f
        self.Q1f = 0.0
        self.Q2f = 0.0
        self.dP1 = 0.0
        self.dP2 = 0.0

        # filter coefficients
        self.aP = math.exp(-2 * math.pi * 40.0 * self.dt)   # pressure LPF ~40 Hz
        self.aQ = math.exp(-2 * math.pi * 25.0 * self.dt)   # flow LPF ~25 Hz

        # homing / bias-calibration bookkeeping
        self._home_timer = 0.0
        self._home_target = None
        self._last_dir = 0     # sign of last clear travel (picks the seated stop)
        self._traverse = 0.0   # net displacement during the current off-stop excursion
        self.q_eps = 0.01 * cfg.sens.q_fullscale   # velocity-blend regulariser

    def step(self, Q1m, Q2m, P1m, P2m, Tm):
        cfg = self.cfg
        dt = self.dt

        if not self._init:
            self.P1f = P1m
            self.P2f = P2m
            self.P1f_prev = P1m
            self.P2f_prev = P2m
            self._init = True

        # ---- filter measurements ----
        self.P1f_prev, self.P2f_prev = self.P1f, self.P2f
        self.P1f = self.aP * self.P1f + (1 - self.aP) * P1m
        self.P2f = self.aP * self.P2f + (1 - self.aP) * P2m
        self.Q1f = self.aQ * self.Q1f + (1 - self.aQ) * Q1m
        self.Q2f = self.aQ * self.Q2f + (1 - self.aQ) * Q2m

        dP1 = (self.P1f - self.P1f_prev) / dt
        dP2 = (self.P2f - self.P2f_prev) / dt
        self.dP1, self.dP2 = dP1, dP2

        # ---- temperature-dependent fluid properties (ablatable) ----
        Teff = Tm if self.use_temp else 20.0
        if self.use_beta_p:
            beta = cfg.fluid.beta_eff(Teff, 0.5 * (self.P1f + self.P2f))
        else:
            beta = cfg.fluid.beta(Teff)
        Cleak = cfg.fluid.leak_coeff(Teff)
        q_leak = Cleak * (self.P1f - self.P2f) if self.use_leak else 0.0

        V1 = max(cfg.cyl.vol_cap(self.x), 1e-6)
        V2 = max(cfg.cyl.vol_rod(self.x), 1e-6)
        comp1 = (V1 / beta) * dP1 if self.use_compress else 0.0
        comp2 = (V2 / beta) * dP2 if self.use_compress else 0.0

        # ---- velocity from each chamber's continuity ----
        v1 = ((Q1m - self.b1) - comp1 - q_leak) / self.A1
        v2 = (comp2 - (Q2m - self.b2) - q_leak) / self.A2

        if self.single_chamber:
            v = v1
        else:
            # SNR-weighted blend (regularised so v->0 when both flows ~0)
            w1 = abs(Q1m)
            w2 = abs(Q2m)
            v = (w1 * v1 + w2 * v2) / (w1 + w2 + self.q_eps)

        # ---- integrate position ----
        self.x += v * dt
        self.x = min(max(self.x, -0.02), cfg.cyl.stroke + 0.02)

        # Latch travel direction (used to identify which stop a seat is against).
        # Accumulate displacement only while the piston is clearly moving (flow
        # above a threshold); the sign is committed once an excursion exceeds 30%
        # of the stroke.  At a stop the excursion resets, so dwell jitter -- which
        # can flip the instantaneous velocity sign, especially at high temperature
        # where leakage corrupts the velocity estimate -- never flips the latch.
        qfs = cfg.sens.q_fullscale
        moving = abs(self.Q1f) > 0.05 * qfs or abs(self.Q2f) > 0.05 * qfs
        if moving:
            self._traverse += v * dt
            if abs(self._traverse) > 0.30 * cfg.cyl.stroke:
                self._last_dir = 1 if self._traverse > 0 else -1
        else:
            self._traverse = 0.0

        # ---- end-stop homing + bias calibration ----
        self._maybe_home(Q1m, Q2m, q_leak)

        # ---- force from calibrated pressure transducers ----
        F_net = P1m * self.A1 - P2m * self.A2

        return self.x, F_net

    def _maybe_home(self, Q1m, Q2m, q_leak):
        cfg = self.cfg
        Ps = cfg.circ.supply_pressure
        qfs = cfg.sens.q_fullscale

        seated = (abs(self.Q1f) < 0.03 * qfs and abs(self.Q2f) < 0.03 * qfs
                  and max(self.P1f, self.P2f) > 0.70 * Ps)

        if not seated:
            self._home_target = None
            self._home_timer = 0.0
            return

        # Which stop is the piston seated against?  The pressure pattern does NOT
        # answer this reliably: under open-loop excitation both chambers can
        # dead-head near supply at *either* stop, and with unequal areas (A1>A2)
        # the force balance then always reads "extended".  The robust, sensorless
        # cue is the direction of travel -- the piston reaches whichever stop it
        # was last moving toward (self._last_dir, latched from the integrated
        # velocity sign).  Until any clear motion is seen, fall back to the side
        # the current estimate is nearer.
        if self._last_dir > 0:
            target = cfg.cyl.stroke
        elif self._last_dir < 0:
            target = 0.0
        else:
            target = cfg.cyl.stroke if self.x > 0.5 * cfg.cyl.stroke else 0.0
        if self._home_target == target:
            self._home_timer += self.dt
        else:
            self._home_target = target
            self._home_timer = 0.0

        if self._home_timer > 0.30:        # debounce 0.3 s
            # pin absolute position (homing)
            if self.use_homing:
                self.x = target

            # Calibrate flow-meter biases. The identity Q_true = +/-q_leak holds
            # only in steady state (dP/dt ~ 0); the steady gate rejects transient
            # seats whose compressibility flow would corrupt the bias.
            if self.use_bias_calib:
                steady = (not self.use_steady_gate) or \
                         (abs(self.dP1) < 1.5e5 and abs(self.dP2) < 1.5e5)
                if steady:
                    self.b1 = self.Q1f - q_leak
                    self.b2 = self.Q2f + q_leak
