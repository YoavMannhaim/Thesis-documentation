## Libraries
import numpy as np

## Imported Functions
from RRTStar import RRTStar
import Visualization as vs
import Kinematics_and_Dynamics as kd

# ---------------------- ROBOT PARAMETERS ----------------------

link_lengths = [0.2, 0.55, 0.45, 0.6]  # [m] [Base height, Link 1 length, Link 2 length, Link 3 length]
link_radii   = [0.05, 0.035, 0.03, 0.025]  # [m] Radius (thickness) of each link

joint_limits = [  # [rad] Min and max rotation for each joint
    [-np.pi, np.pi],       # Joint 1: full rotation allowed
    [-np.pi/2, np.pi/2],   # Joint 2: ±90°
    [-np.pi, np.pi]        # Joint 3: full rotation
]

joint_vel_limit = [  # [rad/s] Min and max allowed velocity for each joint
    [-2.0, 2.0],
    [-2.0, 2.0],
    [-2.0, 2.0]
]

joint_acc_limit = [  # [rad/s²] Min and max allowed acceleration for each joint
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-1.0, 1.0]
]

link_mass    = [1.0, 1.0, 1.0, 1.0]  # [kg] Mass of each link (including base height element)
link_inertia = [1.0, 1.0, 1.0, 1.0]  # [kg·m²] Moment of inertia for each link

# Safety factors for motion limits
S_a = 0.3  # Factor applied to maximum acceleration (safety margin)
S_v = 0.2  # Factor applied to maximum velocity (safety margin)

safety_margin = 0.01  # [m] Extra clearance around obstacles to avoid collisions

# ---------------------- GRIPPER & OBJECT PARAMETERS ----------------------

gripper_param = {
    'length': 0.1,      # [m] Length of the gripper
    'width': 0.2,      # [m] Width of the gripper
    'peel_angle_A': 45, # [deg] Peel angle in one direction
    'peel_angle_B': 45  # [deg] Peel angle in opposite direction
}

object_param = {
    'height': 0.3,     # [m] Height of the object
    'width': 0.2,      # [m] Width of the object
    'length': 0.1,     # [m] Length of the object
    'mass': 0.1,        # [kg] Mass of the object
    'r_cm': [0.1, 0.15] # [m] Offset of the object's center of mass relative to gripper center
}

adhesive_param = {  # Adhesive properties for gripping
    'length': 0.1,                # [m] Adhesive strip length
    'width': 0.2,                 # [m] Adhesive strip width
    'thickness': 0.05,            # [m] Adhesive thickness
    'E': 1e6,                      # [Pa] Elastic modulus
    'v': 0.35,                     # Poisson's ratio
    'shear_strength': 0.5e6,       # [Pa] Shear strength of adhesive
    'surface_energy': 36.0,         # [J/m²] Surface energy
    'poisson_ratio': 0.5,            # Unitless
    'strain': 0.5e-6,              # Unitless strain limit
}

angular_energy_friction = 0.2  # Fraction of available energy used for angular motion

# ---------------------- START & GOAL CONFIGURATIONS ----------------------

start_config = [0.1, 0.1, 0.1]               # [rad] Starting joint angles
goal_config  = [np.pi/4, -np.pi/4, np.pi/6]  # [rad] Goal joint angles

# ---------------------- OBSTACLES ----------------------
# Each obstacle has a 'type' and geometry parameters

obstacles = [
    {  # Circular obstacle
        'type': 'circle',
        'center': np.array([0.6, 0.4]),  # [m] X,Y center
        'radius': 0.07                   # [m] Radius
    },
    {  # Rectangular obstacle
        'type': 'rectangle',
        'bottom_left': np.array([0.8, 0.7]),  # [m] X,Y of bottom-left corner
        'width': 0.1,                         # [m] Width
        'height': 0.3                         # [m] Height
    }
]

# ---------------------- PLANNER PARAMETERS ----------------------

planner = RRTStar(
    start_config=start_config,            # Starting joint configuration
    goal_config=goal_config,              # Target joint configuration
    joint_limits=joint_limits,            # Joint angle limits
    joint_vel_limit=joint_vel_limit,      # Max/min joint velocities
    joint_acc_limit=joint_acc_limit,      # Max/min joint accelerations
    link_lengths=link_lengths,            # Link lengths
    link_radii=link_radii,                # Link radii (for collision checking)
    safety_margin=safety_margin,          # Clearance margin from obstacles
    obstacles=obstacles,                  # List of environment obstacles
    S_a=S_a,                               # Safety factor for acceleration
    S_v=S_v,                               # Safety factor for velocity
    mass=link_mass,                        # Link masses
    inertia=link_inertia,                  # Link inertias
    gripper_param=gripper_param,           # Gripper geometry
    object_param=object_param,             # Object properties
    adhesive_param=adhesive_param,         # Adhesive properties
    angular_energy_friction=angular_energy_friction,  # Angular energy fraction
    step_size=0.05,                         # [rad] Max expansion step size
    goal_tolerance=0.05,                   # [rad] Allowed final joint-space error
    max_iterations=5000,                   # Max iterations before giving up
    rewire_radius=0.3,                     # [rad] Radius for tree rewiring
    w_time=0.2,                            # Cost weight for time
    w_dist=0.5                             # Cost weight for distance
)

# ---------------------- RUN PLANNER ----------------------
pos,  angle = kd.forward_kinematics(goal_config, link_lengths)
print(f"location of goal in XY: {pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}")
path = planner.plan()  # Run the RRT* planning algorithm

# ---------------------- OUTPUT RESULTS ----------------------

if path:
    print("\nPath found:")
    for i, config in enumerate(path):
        print(f"Step {i}: {np.round(config, 3)}")

    vs.plot_rrt_tree(planner, link_lengths, obstacles)  # assuming you have this function elsewhere
    vs.animate_arm(path, link_lengths, obstacles)  # FK will be used internally inside animate_arm without passing
else:
    print("No path found.")
