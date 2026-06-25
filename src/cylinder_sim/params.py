"""Physical, fluid, sensor and estimator parameters for the sensorless
double-acting hydraulic cylinder simulation.

All quantities are SI unless noted:
    length  [m]      area   [m^2]    volume   [m^3]
    pressure[Pa]     flow   [m^3/s]  force    [N]
    time    [s]      temp   [degC]   mass     [kg]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Cylinder geometry (representative medium-duty industrial actuator)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cylinder:
    bore: float = 0.100          # piston / bore diameter      [m]
    rod: float = 0.056           # rod diameter                [m]
    stroke: float = 1.000        # usable stroke length        [m]
    dead_vol_cap: float = 2.0e-4   # cap-side port+hose dead volume   [m^3]
    dead_vol_rod: float = 2.0e-4   # rod-side port+hose dead volume   [m^3]
    mass: float = 200.0          # moving mass (piston+rod+load) [kg]
    stop_stiffness: float = 1.0e8   # end-stop contact stiffness  [N/m]
    stop_damping: float = 4.0e4     # end-stop contact damping    [N*s/m]

    @property
    def area_cap(self) -> float:
        """Full-bore (cap-side, chamber 1) piston area."""
        return math.pi / 4.0 * self.bore ** 2

    @property
    def area_rod(self) -> float:
        """Annulus (rod-side, chamber 2) piston area."""
        return math.pi / 4.0 * (self.bore ** 2 - self.rod ** 2)

    def vol_cap(self, x: float) -> float:
        """Cap-side chamber volume at position x (x=0 fully retracted)."""
        return self.dead_vol_cap + self.area_cap * x

    def vol_rod(self, x: float) -> float:
        """Rod-side chamber volume at position x."""
        return self.dead_vol_rod + self.area_rod * (self.stroke - x)


# --------------------------------------------------------------------------
# Hydraulic circuit (supply / tank / valve)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Circuit:
    supply_pressure: float = 210.0e5   # pump / supply pressure   [Pa] (210 bar)
    tank_pressure: float = 1.0e5       # return / tank pressure   [Pa] (1 bar)
    # Valve flow gain: Q = kv * u * sqrt(dP). Sized for rated flow at
    # full spool (u=1) across a nominal land pressure drop.
    rated_flow: float = 100.0 / 60.0e3   # 100 L/min -> m^3/s
    nominal_dp: float = 35.0e5           # nominal land dP at rating [Pa]
    valve_bw: float = 50.0               # spool first-order bandwidth [Hz]

    @property
    def kv(self) -> float:
        return self.rated_flow / math.sqrt(self.nominal_dp)


# --------------------------------------------------------------------------
# Fluid with temperature dependence (ISO VG46 mineral hydraulic oil)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fluid:
    beta_ref: float = 1.5e9      # effective bulk modulus @ T_ref [Pa]
    beta_tempco: float = -0.005  # fractional d(beta)/dT  [1/degC]
    T_ref: float = 20.0          # reference temperature  [degC]
    mu_ref: float = 0.0410       # dynamic viscosity @ 40 degC [Pa*s]
    mu_T_ref: float = 40.0       # viscosity reference temp    [degC]
    mu_vogel_B: float = 3100.0   # Vogel-like exponent constant [K]
    leak_coeff_ref: float = 2.0e-13  # cross-piston laminar leak @ 40 degC
    #                                  [m^3/s/Pa], scales with 1/viscosity
    air_fraction: float = 0.02   # entrained-air volume fraction @ atmospheric
    air_polytropic: float = 1.0  # polytropic exponent (1 isothermal .. 1.4 adiab.)
    P_air_ref: float = 1.0e5     # reference (atmospheric) pressure [Pa]

    def beta(self, T: float) -> float:
        """Effective bulk modulus vs temperature. Stiffer (higher) when cold."""
        b = self.beta_ref * (1.0 + self.beta_tempco * (T - self.T_ref))
        return max(b, 0.3e9)

    def beta_eff(self, T: float, P: float) -> float:
        """Effective bulk modulus including entrained air, which expands at low
        pressure and collapses the stiffness. Standard isothermal-air model:
        the air term vanishes at high P (b->beta(T)) and dominates at low P."""
        b = self.beta(T)
        P = max(P, 2.0e5)
        n = self.air_polytropic
        eps = self.air_fraction
        ratio = (self.P_air_ref / P) ** (1.0 / n)
        num = 1.0 + eps * ratio
        den = 1.0 + eps * ratio * b / (n * P)
        return max(b * num / den, 0.1e9)

    def mu(self, T: float) -> float:
        """Dynamic viscosity. Rises sharply when cold (Vogel-like)."""
        Tk = T + 273.15
        Tk_ref = self.mu_T_ref + 273.15
        return self.mu_ref * math.exp(self.mu_vogel_B * (1.0 / Tk - 1.0 / Tk_ref))

    def leak_coeff(self, T: float) -> float:
        """Cross-piston leakage coefficient (more leak when hot/thin)."""
        return self.leak_coeff_ref * (self.mu_ref / self.mu(T))


# --------------------------------------------------------------------------
# Friction (Coulomb + temperature-dependent viscous)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Friction:
    coulomb: float = 600.0       # Coulomb friction force [N]
    stribeck_vel: float = 0.01   # tanh smoothing velocity [m/s]
    viscous_ref: float = 4.0e3   # viscous coeff @ 40 degC [N*s/m]

    def force(self, v: float, mu_ratio: float) -> float:
        """Friction force opposing velocity v. mu_ratio = mu(T)/mu_ref."""
        coul = self.coulomb * math.tanh(v / self.stribeck_vel)
        visc = self.viscous_ref * mu_ratio * v
        return coul + visc


# --------------------------------------------------------------------------
# Sensors (realistic noise / bias / quantization)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sensors:
    sample_rate: float = 1000.0        # sampling / estimator rate [Hz]

    # Pressure transducers
    p_fullscale: float = 250.0e5       # transducer full scale  [Pa]
    p_noise_frac: float = 0.005        # white noise, fraction of FS (1 sigma)
    p_bias_frac: float = 0.0025        # per-run constant bias, fraction of FS (1 sigma)
    p_adc_bits: int = 12               # ADC resolution

    # Flow meters (gear/turbine) -- bias is what wrecks pure integration
    q_fullscale: float = 160.0 / 60.0e3  # 160 L/min -> m^3/s
    q_noise_frac: float = 0.010        # white noise, fraction of reading (1 sigma)
    q_noise_floor_frac: float = 0.002  # white noise floor, fraction of FS
    q_bias_frac: float = 0.005         # per-run constant bias, fraction of FS (1 sigma)

    # Temperature sensor
    T_noise: float = 1.0               # white noise (1 sigma) [degC]
    T_bias: float = 0.5                # per-run bias (1 sigma) [degC]

    # ---- transducer dynamics / non-idealities ----
    p_bandwidth: float = 500.0         # pressure sensor 1st-order bandwidth [Hz]
    q_bandwidth: float = 80.0          # flow meter bandwidth                [Hz]
    delay_ms: float = 1.5              # acquisition/transport delay         [ms]
    bias_drift_frac: float = 0.0015    # slow bias random-walk per second, frac FS
    q_kfactor_nl: float = 0.02         # flow K-factor nonlinearity (frac at FS)
    clip_to_fullscale: bool = True     # flow meter saturates at +/- FS


# --------------------------------------------------------------------------
# Plant / estimator parameter mismatch (1-sigma fractional, applied to the
# Hopsan plant per run; the observer keeps nominal values -> model mismatch)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    beta: float = 0.10       # bulk modulus
    leak: float = 0.20       # cross-piston leakage coefficient
    visc: float = 0.20       # viscous friction
    area: float = 0.005      # piston areas (machined geometry, known well)
    deadvol: float = 0.12    # chamber dead volumes
    enabled: bool = True


# --------------------------------------------------------------------------
# Master configuration bundle
# --------------------------------------------------------------------------


@dataclass
class Config:
    cyl: Cylinder = field(default_factory=Cylinder)
    circ: Circuit = field(default_factory=Circuit)
    fluid: Fluid = field(default_factory=Fluid)
    fric: Friction = field(default_factory=Friction)
    sens: Sensors = field(default_factory=Sensors)
    mismatch: Mismatch = field(default_factory=Mismatch)

    sim_dt: float = 1.0e-4       # truth integration step [s]
    duration: float = 32.0       # trajectory duration    [s]

    @property
    def force_fullscale(self) -> float:
        """Force full-scale = max bore-side force at supply pressure."""
        return self.cyl.area_cap * self.circ.supply_pressure

    @property
    def decimation(self) -> int:
        """Truth steps per sensor sample."""
        return int(round(1.0 / (self.sens.sample_rate * self.sim_dt)))


# Acceptance targets from the requirement
POS_ERR_LIMIT = 0.05    # +/- 5 cm
FORCE_ERR_LIMIT_FRAC = 0.12   # +/- 12 % of full scale
TEMP_RANGE = (0.0, 50.0)      # degC
