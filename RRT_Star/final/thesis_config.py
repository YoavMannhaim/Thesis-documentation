"""
Material and mechanical parameters derived from the thesis
(Yoav_Master_Thesis_2, Chapter 5 - Adhesive Force Based Acceleration Calculation).

Each value below is commented with its thesis source where one exists.
Values not covered by the thesis are kept at the original Run_RRTStar.py
defaults and marked accordingly, since the thesis doesn't report an
equivalent measurement for them.
"""
import numpy as np

# ---------------------- ROBOT PARAMETERS ----------------------
# Robot geometry not covered by the thesis -- retained from the original Run_RRTStar.py
link_lengths = [0.2, 0.55, 0.45, 0.6]      # [m] Base height, Link1, Link2, Link3
link_radii   = [0.05, 0.035, 0.03, 0.025]  # [m] Radius of each link

joint_limits = [
    [-np.pi, np.pi],
    [-np.pi/2, np.pi/2],
    [-np.pi, np.pi],
]
joint_vel_limit = [[-2.0, 2.0]] * 3   # [rad/s]
joint_acc_limit = [[-1.0, 1.0]] * 3   # [rad/s^2]

link_mass    = [1.0, 1.0, 1.0, 1.0]   # [kg]
link_inertia = [1.0, 1.0, 1.0, 1.0]   # [kg m^2]

S_a = 0.3   # Acceleration safety factor
S_v = 0.2   # Velocity safety factor
safety_margin = 0.01  # [m]

# ---------------------- GRIPPER & OBJECT ----------------------
# Gripper contact patch sized to the 3M 1900 tape width used throughout
# the thesis experiments (b = 50 mm, Table 5.5 / section 5.3.1).
gripper_param = {
    'length': 0.1,       # [m] contact length
    'width': 0.05,       # [m] THESIS: 50 mm tape width
    'peel_angle_A': 90,  # [deg] THESIS: 90 deg peel geometry (section 5.3.4.1)
    'peel_angle_B': 90,  # [deg]
}

object_param = {
    'height': 0.3,
    'width': 0.05,       # matched to tape width
    'length': 0.1,
    'mass': 0.1,         # [kg]
    'r_cm': [0.1, 0.15],
}

# ---------------------- ADHESIVE (3M Value Duct Tape 1900) ----------------------
adhesive_param = {
    'length': 0.1,             # [m] bonded strip length
    'width': 0.05,             # [m] THESIS: 50 mm tape width (Table 5.5)
    'thickness': 6.0e-5,       # [m] THESIS: adhesive layer a = 0.060 mm (section 5.3.2)
    'E': 1.5e6,                # [Pa] THESIS: ADHESIVE layer modulus, derived from the
                               #      adopted dynamic shear modulus G = 0.50 MPa
                               #      (sec 5.3.3 Conclusions) via E = 2G(1+v) with v = 0.5.
    'E_backing': 150e6,        # [Pa] THESIS: BACKING modulus E = 150 MPa (Eq. 5.101 worked example)
    'backing_thickness': 1.02e-4,  # [m] THESIS: 2h = 0.102 mm (sec 5.3.2)
    'v': 0.5,
    'poisson_ratio': 0.5,      # THESIS: incompressible elastomer assumption
    'shear_modulus': 0.5e6,    # [Pa] THESIS: adopted dynamic G = 0.50 MPa (section 5.3.3 Conclusions)
    'surface_energy': 288.2,   # [J/m^2] THESIS: uncoated cardboard, Table 5.9 mean
    'shear_strength': 0.5e6,   # [Pa] not in thesis -- see note below
    'strain': 0.5e-6,          # [-]  not in thesis -- retained default
}

# Note on shear_strength: the thesis shear test (section 5.3.3) measured the
# shear modulus (G), not a shear failure strength -- the bond crept under
# sustained load rather than failing outright, so no ultimate shear stress
# was recorded. The original placeholder 0.5e6 Pa is kept here; worth
# revisiting if a real failure-strength measurement becomes available.

# Surface energy for the coated-cardboard case (Table 5.10 mean), available as an
# alternative scenario parameter:
SURFACE_ENERGY_COATED = 471.4  # [J/m^2]

angular_energy_friction = 0.2

# ---------------------- START & GOAL ----------------------
start_config = [0.1, 0.1, 0.1]
goal_config  = [np.pi/4, -np.pi/4, np.pi/6]

# ---------------------- PLANNER ----------------------
planner_params = dict(
    step_size=0.05,
    goal_tolerance=0.05,
    max_iterations=1500,
    rewire_radius=0.3,
    w_time=0.2,
    w_dist=0.5,
)
