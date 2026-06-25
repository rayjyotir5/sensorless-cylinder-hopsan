"""Well-scaled Extended Kalman Filter for the double-acting cylinder.

Augmented state  xi = [x, v, P1, P2, b1, b2, FL]. Measurements are BOTH chamber
pressures AND both flow meters: z = [P1, P2, Q1, Q2]. Using the flows as
measurements (quasi-steady continuity h_Q = +/-A v + b + leak) gives the strong
velocity/bias observability the pressure channel alone lacks. An end-stop penalty
force is included in the process model so the v=0 information at seatings (which
separates velocity from flow-meter bias) is available to the filter.

Two numerical issues are handled explicitly:
  * Conditioning -- the filter runs in NON-DIMENSIONAL coordinates (each state
    divided by a physical scale so it is O(1)); Jacobians are numerical in scaled
    space, so the covariance no longer spans ~20 orders of magnitude.
  * Stiffness -- the v-P subsystem oscillates near the hydraulic natural
    frequency; the prediction uses semi-implicit (symplectic) Euler sub-steps,
    which are stable there, rather than forward Euler.

Drop-in: step(Q1m,Q2m,P1m,P2m,Tm) -> (x_hat, F_hat); pos_sigma() for 1-sigma.
"""
from __future__ import annotations
import numpy as np
from .params import Config

NX = 7
N_SUB = 4
K_STOP = 1.0e7   # end-stop penalty stiffness [N/m] used inside the model


class CylinderEKF:
    def __init__(self, cfg: Config, force_from_measured: bool = False):
        self.cfg = cfg
        c = cfg.cyl
        self.A1, self.A2, self.m = c.area_cap, c.area_rod, c.mass
        self.s_l, self.V01, self.V02 = c.stroke, c.dead_vol_cap, c.dead_vol_rod
        self.dt = 1.0 / cfg.sens.sample_rate
        self.force_from_measured = force_from_measured

        Ps = cfg.circ.supply_pressure
        qfs = cfg.sens.q_fullscale
        self.s_P, self.s_Q = Ps, qfs
        # state scales: x,v,P1,P2,b1,b2,FL
        self.D = np.array([c.stroke, 0.3, Ps, Ps, 0.02*qfs, 0.02*qfs,
                           cfg.force_fullscale])

        self.xs = np.array([0.0, 0.0, cfg.circ.tank_pressure/Ps,
                            cfg.circ.tank_pressure/Ps, 0.0, 0.0, 0.0])
        # initial covariance: bias is only moderately uncertain (true offset is
        # small), so the first ramp is attributed mostly to velocity not bias
        self.P = np.diag([0.02**2, 0.2**2, 0.01**2, 0.01**2, 0.3**2, 0.3**2, 0.4**2])
        # bias nearly frozen between seats (held, like the observer's calibration)
        self.Q = np.diag([1e-4**2, 0.03**2, 0.004**2, 0.004**2,
                          1e-5**2, 1e-5**2, 3e-3**2])
        sigP = cfg.sens.p_noise_frac * cfg.sens.p_fullscale / Ps
        sigQ = 0.05            # scaled flow-meas std (incl. quasi-steady model error)
        self.R = np.diag([(1.8*sigP)**2, (1.8*sigP)**2, sigQ**2, sigQ**2])

        self.Fc, self.vs, self.Bv = cfg.fric.coulomb, cfg.fric.stribeck_vel, cfg.fric.viscous_ref
        self._seat_cnt = 0
        self._seat_target = None
        self.use_homing = True   # apply seated x/v pseudo-measurements (EKF homing)
        # filtered pressures for the compressibility term in the flow measurement
        self.P1f = cfg.circ.tank_pressure
        self.P2f = cfg.circ.tank_pressure
        self.aP = np.exp(-2*np.pi*40.0*self.dt)
        self._dP1 = 0.0
        self._dP2 = 0.0

    def _stop_force(self, x, v):
        if x < 0.0:
            return K_STOP * (-x) - 2e4 * min(v, 0.0)
        if x > self.s_l:
            return -K_STOP * (x - self.s_l) - 2e4 * max(v, 0.0)
        return 0.0

    def _substep(self, xi, Q1m, Q2m, beta, Cleak, h):
        x, v, P1, P2, b1, b2, FL = xi
        V1 = max(self.V01 + self.A1 * x, 1e-6)
        V2 = max(self.V02 + self.A2 * (self.s_l - x), 1e-6)
        qleak = Cleak * (P1 - P2)
        Ffric = self.Fc * np.tanh(v / self.vs) + self.Bv * v
        Fstop = self._stop_force(x, v)
        v_new = v + h * (P1*self.A1 - P2*self.A2 - Ffric - FL + Fstop) / self.m
        P1_new = P1 + h * (beta/V1) * ((Q1m - b1) - self.A1*v_new - qleak)
        P2_new = P2 + h * (beta/V2) * ((Q2m - b2) + self.A2*v_new + qleak)
        x_new = x + h * v_new
        return np.array([x_new, v_new, P1_new, P2_new, b1, b2, FL])

    def _f_scaled(self, xs, Q1m, Q2m, beta, Cleak):
        xi = xs * self.D
        h = self.dt / N_SUB
        for _ in range(N_SUB):
            xi = self._substep(xi, Q1m, Q2m, beta, Cleak, h)
        return xi / self.D

    def _h_scaled(self, xs, beta, Cleak):
        """Predicted measurements [P1,P2,Q1,Q2] in scaled units. The flow model
        includes the compressibility term (V/beta) dP from the measured pressure
        derivative, so velocity is not biased during fast pressure transients."""
        xi = xs * self.D
        x, v, P1, P2, b1, b2, FL = xi
        V1 = max(self.V01 + self.A1 * x, 1e-6)
        V2 = max(self.V02 + self.A2 * (self.s_l - x), 1e-6)
        qleak = Cleak * (P1 - P2)
        Q1p = self.A1*v + b1 + qleak + (V1/beta)*self._dP1
        Q2p = -self.A2*v + b2 - qleak + (V2/beta)*self._dP2
        return np.array([P1/self.s_P, P2/self.s_P, Q1p/self.s_Q, Q2p/self.s_Q])

    def step(self, Q1m, Q2m, P1m, P2m, Tm):
        beta = self.cfg.fluid.beta(Tm)
        Cleak = self.cfg.fluid.leak_coeff(Tm)
        eps = 1e-6
        # measured pressure derivative (filtered) for the compressibility term
        P1f0, P2f0 = self.P1f, self.P2f
        self.P1f = self.aP*self.P1f + (1-self.aP)*P1m
        self.P2f = self.aP*self.P2f + (1-self.aP)*P2m
        self._dP1 = (self.P1f - P1f0) / self.dt
        self._dP2 = (self.P2f - P2f0) / self.dt

        # ---- predict ----
        f0 = self._f_scaled(self.xs, Q1m, Q2m, beta, Cleak)
        F = np.empty((NX, NX))
        for k in range(NX):
            xp = self.xs.copy(); xp[k] += eps
            F[:, k] = (self._f_scaled(xp, Q1m, Q2m, beta, Cleak) - f0) / eps
        self.xs = f0
        self.P = F @ self.P @ F.T + self.Q

        # ---- update with [P1,P2,Q1,Q2] ----
        z = np.array([P1m/self.s_P, P2m/self.s_P, Q1m/self.s_Q, Q2m/self.s_Q])
        h0 = self._h_scaled(self.xs, beta, Cleak)
        H = np.empty((4, NX))
        for k in range(NX):
            xp = self.xs.copy(); xp[k] += eps
            H[:, k] = (self._h_scaled(xp, beta, Cleak) - h0) / eps
        y = z - h0
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.xs = self.xs + K @ y
        I_KH = np.eye(NX) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # ---- end-stop homing pseudo-measurements: [x, v] = [stop, 0] ----
        # Absolute position is unobservable from P/Q alone; a seated stop fixes
        # it (x known) and v=0 there separates velocity from flow-meter bias.
        Ps, qfs = self.s_P, self.s_Q
        seated = (abs(Q1m) < 0.03*qfs and abs(Q2m) < 0.03*qfs and
                  max(P1m, P2m) > 0.70*Ps)
        target = self.s_l if P1m > P2m else 0.0
        if seated and self._seat_target == target:
            self._seat_cnt += 1
        else:
            self._seat_target = target if seated else None
            self._seat_cnt = 0
        if self.use_homing and seated and self._seat_cnt * self.dt > 0.20:
            Hx = np.zeros((2, NX)); Hx[0, 0] = 1.0; Hx[1, 1] = 1.0
            zx = np.array([target/self.D[0], 0.0])             # x=target, v=0 (scaled)
            Rx = np.diag([(0.005/self.D[0])**2, (0.01/self.D[1])**2])
            yx = zx - Hx @ self.xs
            Sx = Hx @ self.P @ Hx.T + Rx
            Kx = self.P @ Hx.T @ np.linalg.inv(Sx)
            self.xs = self.xs + Kx @ yx
            IKx = np.eye(NX) - Kx @ Hx
            self.P = IKx @ self.P @ IKx.T + Kx @ Rx @ Kx.T

        xi = self.xs * self.D
        x_hat = float(np.clip(xi[0], 0.0, self.s_l))
        F_hat = (P1m*self.A1 - P2m*self.A2) if self.force_from_measured \
            else (xi[2]*self.A1 - xi[3]*self.A2)
        return x_hat, F_hat

    def pos_sigma(self):
        return float(np.sqrt(max(self.P[0, 0], 0.0)) * self.D[0])
