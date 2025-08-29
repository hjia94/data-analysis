#!/usr/bin/env python3
"""
Two-chamber vacuum model with a pinhole between chambers.

- Chamber A gets a gas inflow (e.g., sccm).
- Chamber B is connected to a pump (speed S at the chamber).
- Chambers are connected by a small aperture (pinhole) with molecular-flow conductance C.
- Computes steady-state pressures, particle numbers, and transient evolution.

Units:
- Pressures P in Torr (internally for vacuum throughput math).
- Throughput Q and conductances C, S in Torr*L/s and L/s.
- Volumes V in liters for ODEs (convert from m^3 as needed).
- Temperature T in K.
- For particle numbers, pressures are converted to Pascals and volumes to m^3.

Equations (throughput balance):
  Chamber A:  Q_in = C (P_A - P_B) + V_A * dP_A/dt
  Chamber B:  C (P_A - P_B) = S P_B + V_B * dP_B/dt

Steady-state (dP/dt = 0):
  P_B = Q_in / S
  P_A = P_B + Q_in / C

Transient (matrix form):
  d/dt [P_A, P_B]^T = A [P_A, P_B]^T + b
    where A = [[-C/V_A,  C/V_A],
               [ C/V_B, -(C+S)/V_B]]
          b = [Q_in/V_A, 0]^T
  Let p = P - P_ss, then p' = A p  ==> eigenvalues(A) give time constants tau_i = -1 / Re(lambda_i).

Author: ChatGPT
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.constants import physical_constants, elementary_charge as QE
from scipy.constants import atomic_mass as AMU

from plasma_utils import ion_sound_speed

# ----------------------------- Constants -----------------------------
K_B = 1.380649e-23      # Boltzmann constant [J/K]
TORR_TO_PA = 133.322368 # 1 Torr = 133.322368 Pa
L_TO_M3 = 1e-3          # 1 L = 1e-3 m^3
SCCM_TO_TORR_L_S = 0.0127  # Throughput conversion at ~STP: Q[Torr*L/s] = 0.0127 * sccm

# ----------------------------- Data classes -----------------------------
@dataclass
class PlasmaParams:
    nA_m3: float = 1e15          # electron/ion density in chamber A [m^-3]
    Te_eV: float = 1          # electron temperature [eV]
    mi_amu: float = 16.0  # ion mass
    tau_p: float = 0.5    # transmission factor (0..1), geometry/sheath dependent
    
@dataclass
class Geometry:
    d_A_cm: float = 2.0      # Chamber A diameter [cm]
    d_B_cm: float = 40.0     # Chamber B diameter [cm]
    l_A_cm: float = 25.0     # Chamber A length [cm]
    l_B_cm: float = 105.0    # Chamber B length [cm]

    V_A_m3: float = np.pi * (d_A_cm * 1e-2 / 2)**2 * l_A_cm * 1e-2  # Chamber A volume [m^3]
    V_B_m3: float = np.pi * (d_B_cm * 1e-2 / 2)**2 * l_B_cm * 1e-2  # Chamber B volume [m^3]

    # sheath-collecting wall area in B [m^2] (sum of internal surfaces)
    A_wall_B_cm2: float = 2*np.pi*(d_B_cm/2)**2 + np.pi*d_B_cm * l_B_cm * 1.1
    A_wall_B_m2: float = A_wall_B_cm2 * 1e-4  # Convert to m^2
    
    pinhole_diameter_mm: float = 1      # If using pinhole model (molecular flow)
    A_hole_cm2: float = np.pi * (pinhole_diameter_mm * 1e-2 / 2)**2  # aperture (hole) area between A and B [cm^2]
    A_hole_m2: float = A_hole_cm2 * 1e-4  # Convert to m^2

@dataclass
class SystemParams:
    # Chamber volumes (get from Geometry by default)
    V_A_m3: float = Geometry().V_A_m3
    V_B_m3: float = Geometry().V_B_m3
    
    T_K: float = 300.0         # Temperature [K]
    inflow_sccm: float = 1  # Gas inflow into A [sccm]
    S_A_L_s: float = 100     # NEW: Pump on chamber A (0 means none)
    S_B_L_s: float = 1000.0   # Pump on chamber B
    # Either set conductance_L_s directly, or describe a pinhole to estimate it:
    conductance_L_s: float | None = None  # Directly specify C [L/s] if known

    gas_molar_mass_g_per_mol: float = 32.0  # N2 ~ 28; He ~ 4; Ar ~ 40; O2 ~ 32
    use_pinhole_model: bool = True        # If True, compute C from pinhole instead of using conductance_L_s
    
    # Pinhole parameters (get from Geometry by default)
    pinhole_diameter_mm: float = Geometry().pinhole_diameter_mm

@dataclass
class SimulationParams:
    t_end_s: float = 5.0        # total simulation time [s]
    dt_s: float = 1e-3          # time step [s]
    P_A0_Torr: float = 1e-3     # initial pressure in A [Torr]
    P_B0_Torr: float = 1e-3     # initial pressure in B [Torr]
    enable_plot: bool = False   # requires matplotlib


# ----------------------------- Helpers -----------------------------
def sccm_to_throughput(sccm: float) -> float:
    """Convert gas inflow in sccm to throughput Q in Torr*L/s."""
    return SCCM_TO_TORR_L_S * sccm

def pinhole_conductance_L_s(A_hole_cm2: float, T_K: float = 293.0, gas: str = "O2") -> float:
    """
    Molecular-flow orifice conductance (L/s) using Leybold data.

    Base formula (air/N2, 20°C):
        C = 11.6 * A_cm2

    Gas factor:
        O2: 0.947
        N2/air: 1.0
        (others can be added)

    Temperature scaling ~ sqrt(T/293).
    """

    A_cm2 = A_hole_cm2
    C_ref = 11.6 * A_cm2

    gas_factors = {"N2": 1.0, "air": 1.0, "O2": 0.947}
    f_gas = gas_factors.get(gas, 1.0)

    return C_ref * f_gas * math.sqrt(T_K / 293.0)

def effective_speed_seen_from_A(S_A: float, C: float, S_B: float) -> float:
    # Series of C and S_B: S_series = (C*S_B)/(C+S_B). Parallel with S_A: add them.
    return S_A + (C * S_B) / (C + S_B)

def steady_state_pressures(Q_in: float, S_A: float, S_B: float, C: float) -> tuple[float, float]:
    # P_A = Q_in / ( S_A + (C*S_B)/(C+S_B) );  P_B = (C/(C+S_B)) * P_A
    if C <= 0 or S_B < 0 or S_A < 0:
        raise ValueError("Require C>0 and S_A,S_B >= 0.")
    S_eff = effective_speed_seen_from_A(S_A, C, S_B)
    P_A = Q_in / S_eff
    P_B = (C / (C + S_B)) * P_A
    return P_A, P_B
def particle_number(P_Torr: float, V_m3: float, T_K: float) -> float:
    return (P_Torr * TORR_TO_PA * V_m3) / (K_B * T_K)

def build_system_matrix(V_A_L: float, V_B_L: float, C_L_s: float, S_A_L_s: float, S_B_L_s: float):
    A = np.array([
        [-(C_L_s + S_A_L_s) / V_A_L,   C_L_s / V_A_L],
        [   C_L_s / V_B_L,            -(C_L_s + S_B_L_s) / V_B_L]
    ], dtype=float)
    return A

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
    k3 = f(t + 0.5*dt, y + 0.5*dt*k2)
    k4 = f(t + dt, y + dt*k3)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Design helper: keep P_A constant when changing S_A or S_B by solving for C2.
def C_to_hold_Pa_constant(Q_in: float, P_A_target: float, S_A_new: float, S_B_new: float) -> float:
    # Want P_A_target = Q_in / ( S_A_new + (C*S_B_new)/(C+S_B_new) )
    # Solve for C: let x=C. Define f(x)=S_A_new + (x*S_B_new)/(x+S_B_new) = Q_in/P_A_target
    RHS = Q_in / P_A_target
    a = S_B_new
    b = RHS - S_A_new
    if b <= 0:
        return float('inf')
    denom = (a - b)
    if denom <= 0:
        return float('inf')
    return (b * a) / denom

def solve(params: SystemParams, sim: SimulationParams, geom: Geometry = None, plasma: PlasmaParams = None):
    V_A_L = params.V_A_m3 / L_TO_M3
    V_B_L = params.V_B_m3 / L_TO_M3
    Q_in = sccm_to_throughput(params.inflow_sccm)

    if params.use_pinhole_model or params.conductance_L_s is None:
        C = pinhole_conductance_L_s(params.pinhole_diameter_mm, params.T_K, params.gas_molar_mass_g_per_mol)
        C_source = f"pinhole model (diameter={params.pinhole_diameter_mm} mm, gas M={params.gas_molar_mass_g_per_mol} g/mol)"
    else:
        C = params.conductance_L_s
        C_source = "user-specified"

    P_A_ss, P_B_ss = steady_state_pressures(Q_in, params.S_A_L_s, params.S_B_L_s, C)
    N_A = particle_number(P_A_ss, params.V_A_m3, params.T_K)
    N_B = particle_number(P_B_ss, params.V_B_m3, params.T_K)

    # Calculate plasma densities if parameters are provided
    plasma_results = {}
    if geom is not None and plasma is not None:
        ambipolar_results = estimate_nB_ambipolar(geom, plasma)
        plasma_results = {
            "nA_m3": plasma.nA_m3,
            "nB_m3": ambipolar_results["nB_m3"],
            "ratio_nB_over_nA": ambipolar_results["ratio_nB_over_nA"],
            "c_sA_m_s": ambipolar_results["c_sA_m_s"],
            "c_sB_m_s": ambipolar_results["c_sB_m_s"],
            "C_p_m3_s": ambipolar_results["C_p_m3_s"],
            "L_B_m3_s": ambipolar_results["L_B_m3_s"]
        }

    A = build_system_matrix(V_A_L, V_B_L, C, params.S_A_L_s, params.S_B_L_s)
    b = np.array([Q_in / V_A_L, 0.0], dtype=float)
    evals, _ = np.linalg.eig(A)
    taus = np.array([-1.0/e.real if e.real < 0 else float('inf') for e in evals], dtype=float)

    t_hist = None
    P_hist = None
    if sim.t_end_s > 0 and sim.dt_s > 0:
        def f(t, P):
            return A @ P + b
        steps = int(math.ceil(sim.t_end_s / sim.dt_s))
        t_hist = np.zeros(steps+1)
        P_hist = np.zeros((steps+1, 2))
        P_hist[0] = np.array([sim.P_A0_Torr, sim.P_B0_Torr], dtype=float)
        for i in range(steps):
            t_hist[i+1] = t_hist[i] + sim.dt_s
            P_hist[i+1] = rk4_step(f, t_hist[i], P_hist[i], sim.dt_s)

    S_eff_A = effective_speed_seen_from_A(params.S_A_L_s, C, params.S_B_L_s)

    return {
        "inputs": {
            "V_A_m3": params.V_A_m3, "V_B_m3": params.V_B_m3, "T_K": params.T_K,
            "inflow_sccm": params.inflow_sccm,
            "S_A_L_s": params.S_A_L_s, "S_B_L_s": params.S_B_L_s,
            "C_L_s": C, "C_source": C_source
        },
        "steady_state": {
            "P_A_Torr": P_A_ss, "P_B_Torr": P_B_ss, "N_A": N_A, "N_B": N_B,
            "S_eff_seen_from_A_L_s": S_eff_A
        },
        "plasma": plasma_results,
        "dynamics": {
            "A_matrix": A, "eigenvalues": evals, "time_constants_s": taus,
            "time_history_s": t_hist, "pressure_history_Torr": P_hist
        }
    }

def pretty_print(results, verbose=False):
    ss = results["steady_state"]
    plasma = results.get("plasma", {})
    
    print(f"P_A = {ss['P_A_Torr']:.6g} Torr")
    print(f"P_B = {ss['P_B_Torr']:.6g} Torr")
    print(f"N_A = {ss['N_A']:.6g} molecules")
    print(f"N_B = {ss['N_B']:.6g} molecules")
    
    # Print plasma densities if available
    if plasma:
        print(f"nA = {plasma['nA_m3']:.6g} m^-3")
        print(f"nB = {plasma['nB_m3']:.6g} m^-3")
    
    if verbose:
        print("=== Two-Chamber Vacuum Model (Pinhole Coupling) ===")
        inp = results["inputs"]
        dyn = results["dynamics"]
        print(f"Volumes: V_A = {inp['V_A_m3']} m^3, V_B = {inp['V_B_m3']} m^3")
        print(f"Temperature: T = {inp['T_K']} K")
        print(f"Inflow: {inp['inflow_sccm']} sccm  -> Q_in = {SCCM_TO_TORR_L_S*inp['inflow_sccm']:.6g} Torr·L/s")
        print(f"Pump speed at B: S = {inp['S_L_s']} L/s")
        print(f"Inter-chamber conductance: C = {inp['C_L_s']:.6g} L/s  ({inp['C_source']})")
        print("\n-- Steady State --")
        print(f"P_B = Q_in / S = {ss['P_B_Torr']:.6g} Torr")
        print(f"P_A = P_B + Q_in / C = {ss['P_A_Torr']:.6g} Torr")
        print(f"N_A = {ss['N_A']:.6g} molecules")
        print(f"N_B = {ss['N_B']:.6g} molecules")
        print("\n-- Dynamics --")
        print("A matrix [1/s]:\n", dyn["A_matrix"])
        print("Eigenvalues [1/s]:", dyn["eigenvalues"])
        print("Time constants tau = -1/Re(lambda) [s]:", dyn["time_constants_s"])
        if dyn["time_history_s"] is not None:
            t = dyn["time_history_s"]
            P = dyn["pressure_history_Torr"]
            print(f"\nSimulated {t[-1]:.3g} s with {len(t)} steps.")
            print("Final simulated pressures [Torr]: P_A = %.6g, P_B = %.6g" % (P[-1,0], P[-1,1]))

def estimate_nB_ambipolar(geom: Geometry, plasma: PlasmaParams, TeB_eV: float | None = None) -> dict:
    """
    Ambipolar-leak model (no sources in B):
      C_p = tau_p * A_hole * c_sA      [m^3/s]   (plasma 'conductance' through hole)
      L_B = A_wall_B * c_sB            [m^3/s]   (loss capacity of B walls)
      n_B = (C_p / (C_p + L_B)) * n_A
    """
    c_sA = ion_sound_speed(plasma.Te_eV, plasma.mi_amu)
    c_sB = ion_sound_speed(TeB_eV if TeB_eV is not None else plasma.Te_eV, plasma.mi_amu)

    C_p = max(0.0, plasma.tau_p) * geom.A_hole_m2 * c_sA
    L_B = geom.A_wall_B_m2 * c_sB

    nB = (C_p / (C_p + L_B)) * plasma.nA_m3 if (C_p + L_B) > 0 else 0.0

    return {
        "nB_m3": nB,
        "ratio_nB_over_nA": (nB / plasma.nA_m3) if plasma.nA_m3 > 0 else 0.0,
        "c_sA_m_s": c_sA,
        "c_sB_m_s": c_sB,
        "C_p_m3_s": C_p,
        "L_B_m3_s": L_B
    }
# ----------------------------- Run example -----------------------------
if __name__ == "__main__":
    # Use default parameters defined in the dataclasses above
    # Define plasma parameters for demonstration
    plasma_params = PlasmaParams()
    
    geom_params = Geometry()
    
    results = solve(SystemParams(), SimulationParams(), geom_params, plasma_params)
    pretty_print(results)

    # Optional plotting
    if SimulationParams().enable_plot:
        try:
            import matplotlib.pyplot as plt
            t = results["dynamics"]["time_history_s"]
            P = results["dynamics"]["pressure_history_Torr"]
            if t is not None and P is not None:
                plt.figure()
                plt.plot(t, P[:,0], label="P_A [Torr]")
                plt.plot(t, P[:,1], label="P_B [Torr]")
                plt.xlabel("Time [s]")
                plt.ylabel("Pressure [Torr]")
                plt.legend()
                plt.title("Two-Chamber Pumpdown/Fill Transient")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print("Plotting skipped (matplotlib not available):", e)
