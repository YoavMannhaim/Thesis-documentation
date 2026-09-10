## Libraries
import numpy as np
import math

## Imported Functions
import Max_movement_2D as mm
import Restriction_Checks as rc

## Kinematics and Jacobians of the robot arm

def forward_kinematics(theta, links):
    """
    FK for a 3 joints arm, lifted from ground
    :param theta: Angles of links
    :param links:  Length of links, links[0] - distance of robot from ground
    :return: Position of the end of the three links
    """


    # Base position (elevated by links[0])
    p_0 = np.array([0, links[0]])

    # First joint position
    p_1 = p_0 + links[1] * np.array([np.cos(theta[0]), np.sin(theta[0])])

    # Second joint position (cumulative angle)
    angle_01 = theta[0] + theta[1]
    p_2 = p_1 + links[2] * np.array([np.cos(angle_01), np.sin(angle_01)])

    # Third joint position (cumulative angle)
    angle_02 = angle_01 + theta[2]
    p_3 = p_2 + links[3] * np.array([np.cos(angle_02), np.sin(angle_02)])

    # Build vector pos
    pos = [p_0, p_1, p_2, p_3]

    return pos, angle_02

def inverse_kinematics(position, links, gripper_angle):

    """
    :param position: The gripper position [x,y]
    :param links: Length of links, links[0] - distance of robot from ground
    :param gripper_angle: The angle of the gripper in regard to the base of the robot
    :return: Angles of the 3 joints
    """

    # Parameters
    pos = position
    phi = gripper_angle

    # Calculate location of third joint from the base
    x3 = pos[0] - links[3]*np.cos(phi) # Distance from ground
    y3 = pos[1] - links[3]*np.sin(phi) # Distance from ground
    dx = x3 # Distance from base
    dy = y3 - links[0] # Distance from base
    d = np.sqrt(dx**2+dy**2) # Absolut distance from base

    # Possibility check
    if not (abs(links[1]-links[2]) <= d <= (links[1]+links[2])):
        return None # No solution

    # Calculate angle of Joint 2
    cos2 = (d**2 - links[1]**2 - links[2]**2) / (2 * links[1] * links[2])
    if abs(cos2) > 1:
        return None # No solution

    angle_2 = [np.arccos(cos2), -np.arccos(cos2)]

    angles = [] # Results of the angles in the two possible positions

    for theta_2 in angle_2:
        # Calculate angle of Joint 1
        gamma = np.arctan2(dy, dx)
        delta = np.arctan2(links[2] * np.sin(theta_2), links[1] + links[2] * np.cos(theta_2))
        theta_1 = gamma - delta

        # Calculate angle of Joint 3
        theta_3 = phi - (theta_1 + theta_2)

        angles.append((theta_1, theta_2, theta_3))

    return angles # Two sets of possible angles

def extract_link_segments_from_fk(theta, links):

    """

    :param theta: angles of the Links.
    :param links: Links length.

    :return: Position of links in the world.
    """

    ## Find Position of Joints and define link segments
    pos, gripper_angle = forward_kinematics(theta, links) # Find Position of the 4 joints (base + 3 joints)
    link_seg =[
        (pos[0], pos[1]), # Base to Joint 1
        (pos[1], pos[2]), # Joint 1 to Joint 2
        (pos[2], pos[3]), # Joint 2 to End Effector
    ]

    return pos, link_seg

def Jacobian_2d(theta, links):

    # Trigo calc
    sin1 = np.sin(theta[0])
    sin12 = np.sin(theta[0]+theta[1])
    sin123 = np.sin(theta[0] + theta[1]+theta[2])
    cos1 = np.cos(theta[0])
    cos12 = np.cos(theta[0]+theta[1])
    cos123 = np.cos(theta[0] + theta[1]+theta[2])

    J = np.array([
        [-links[1] * sin1 -links[2] * sin12 -links[3] * sin123, -links[2] * sin12 -links[3] * sin123, -links[3] * sin123],
        [links[1] * cos1 + links[2] * cos12 + links[3] * cos123, links[2] * cos12 + links[3] * cos123, links[3] * cos123],
        [1, 1, 1]
        ])

    return J

def Jacobian_dot_2d(theta, theta_dot, links):

    """

    :param theta: Angle of all 3 Joints
    :param theta_dot: Angular velocity of all 3 Joints
    :param links: Length of all 4 Links [ Base height, Link 1, Link 2, Link 3]

    :return: The derivative of the Jacobian by time
    """

    ## Useful calculations
    sin1 = np.sin(theta[0])
    sin12 = np.sin(theta[0] + theta[1])
    sin123 = np.sin(theta[0] + theta[1] + theta[2])
    cos1 = np.cos(theta[0])
    cos12 = np.cos(theta[0] + theta[1])
    cos123 = np.cos(theta[0] + theta[1] + theta[2])
    theta_dot_scalar = [float(np.asarray(td).reshape(-1)[0]) if hasattr(td, '__iter__') else float(td) for td in theta_dot] # Changes theta dot from list to scalar

    ## Calculation of each individual part of each integration, by q_dot 1-3

    #J11
    dJ11_1 = -links[1] * cos1 - links[2] * cos12 - links[3] * cos123
    dJ11_2 = -links[2] * cos12 - links[3] * cos123
    dJ11_3 = -links[3] * cos123

    #J12
    dJ12_1 = dJ12_2 =- links[2] * cos12 - links[3] * cos123
    dJ12_3 = -links[3] * cos123

    #J13
    dJ13_1 = dJ13_2 = dJ13_3 = -links[3] * cos123

    #J21
    dJ21_1 = -links[1] * sin1 - links[2] * sin12 - links[3] * sin123
    dJ21_2 = -links[2] * sin12 - links[3] * sin123
    dJ21_3 = -links[3] * sin123

    #J22
    dJ22_1 = dJ22_2 = -links[2] * sin12 - links[3] * sin123
    dJ22_3 = -links[3] * sin123

    #J23
    dJ23_1 = dJ23_2 = dJ23_3 = -links[3] * sin123

    ## Calculation of each cell in the array

    #dJ1
    dJ11 = dJ11_1 * theta_dot_scalar[0] + dJ11_2 * theta_dot_scalar[1] + dJ11_3 * theta_dot_scalar[2]
    dJ12 = dJ12_1 * theta_dot_scalar[0] + dJ12_2 * theta_dot_scalar[1] + dJ12_3 * theta_dot_scalar[2]
    dJ13 = dJ13_1 * theta_dot_scalar[0] + dJ13_2 * theta_dot_scalar[1] + dJ13_3 * theta_dot_scalar[2]

    #dJ2
    dJ21 = dJ21_1 * theta_dot_scalar[0] + dJ21_2 * theta_dot_scalar[1] + dJ21_3 * theta_dot_scalar[2]
    dJ22 = dJ22_1 * theta_dot_scalar[0] + dJ22_2 * theta_dot_scalar[1] + dJ22_3 * theta_dot_scalar[2]
    dJ23 = dJ23_1 * theta_dot_scalar[0] + dJ23_2 * theta_dot_scalar[1] + dJ23_3 * theta_dot_scalar[2]

    #dJ3
    dJ31 = dJ32 = dJ33 = 0

    ## Calculate J_dot

    J_dot = np.array([
        [dJ11, dJ12, dJ13],
        [dJ21, dJ22, dJ23],
        [dJ31, dJ32, dJ33]
    ])

    return J_dot

def inertia_matrix(theta, links, mass, inertia):
    """
    Calculates the Inertia Matrix for the Dynamic Equation of Tau = M(q)*qdd + C(q,qd)*qd + G(q)
    :param theta: Angles of the 3 Joints [theta1, theta2, theta3]
    :param links: The length of the links, links[0] - distance from ground before first Joint
    :param mass: The masses of every link [m0, m1, m2, m3] or [m1, m2, m3]
    :param inertia: The moment of inertia for each link about its center of mass [I0, I1, I2, I3] or [I1, I2, I3]
    :return: The 3x3 Inertia Matrix M(q)
    """

    # Handle different indexing conventions
    if len(mass) == 4:  # [m0, m1, m2, m3] - includes base mass
        m1, m2, m3 = mass[1], mass[2], mass[3]
        I1, I2, I3 = inertia[1], inertia[2], inertia[3]
    else:  # [m1, m2, m3] - only link masses
        m1, m2, m3 = mass[0], mass[1], mass[2]
        I1, I2, I3 = inertia[0], inertia[1], inertia[2]

    # Link lengths
    L1, L2, L3 = links[1], links[2], links[3]

    # Center of mass distances (assuming COM at center of each link)
    lc1, lc2, lc3 = L1 / 2, L2 / 2, L3 / 2

    # Trigonometric calculations
    cos2 = np.cos(theta[1])
    cos3 = np.cos(theta[2])
    cos23 = np.cos(theta[1] + theta[2])

    # Initialize Inertia Matrix
    M = np.zeros([3, 3])

    # Element M11 (inertia seen by joint 1)
    M[0, 0] = (I1 + I2 + I3 +
               m1 * lc1 ** 2 +
               m2 * (L1 ** 2 + lc2 ** 2 + 2 * L1 * lc2 * cos2) +
               m3 * (L1 ** 2 + L2 ** 2 + lc3 ** 2 + 2 * L1 * L2 * cos2 +
                     2 * L1 * lc3 * cos23 + 2 * L2 * lc3 * cos3))

    # Element M12 = M21 (coupling between joints 1 and 2)
    # Fixed: removed incorrect /2 factors
    M[0, 1] = M[1, 0] = (I2 + I3 +
                         m2 * (lc2 ** 2 + L1 * lc2 * cos2) +
                         m3 * (L2 ** 2 + lc3 ** 2 + L1 * L2 * cos2 +
                               L1 * lc3 * cos23 + L2 * lc3 * cos3))

    # Element M13 = M31 (coupling between joints 1 and 3)
    # Fixed: removed incorrect /2 factors
    M[0, 2] = M[2, 0] = (I3 +
                         m3 * (lc3 ** 2 + L1 * lc3 * cos23 + L2 * lc3 * cos3))

    # Element M22 (inertia seen by joint 2)
    M[1, 1] = (I2 + I3 +
               m2 * lc2 ** 2 +
               m3 * (L2 ** 2 + lc3 ** 2 + 2 * L2 * lc3 * cos3))

    # Element M23 = M32 (coupling between joints 2 and 3)
    # Fixed: removed incorrect /2 factor
    M[1, 2] = M[2, 1] = (I3 +
                         m3 * (lc3 ** 2 + L2 * lc3 * cos3))

    # Element M33 (inertia seen by joint 3)
    M[2, 2] = I3 + m3 * lc3 ** 2

    return M

def calculate_effective_inertias(theta, links, mass, inertia, acceleration_direction):
    """
    Calculate both linear and angular effective inertias for max_velocity_2D function

    :param theta: Current joint angles [θ1, θ2, θ3]
    :param links: Link lengths [base_height, L1, L2, L3]
    :param mass: Link masses [m1, m2, m3] or [m0, m1, m2, m3]
    :param inertia: Link inertias [I1, I2, I3] or [I0, I1, I2, I3]
    :param acceleration_direction: Desired motion direction in rad

    :return:
    I_eff_linear: Effective inertia for linear motion
    I_eff_angular: Effective inertia for angular motion
    """

    # Calculate the configuration-dependent inertia matrix
    M = inertia_matrix(theta, links, mass, inertia)

    # Calculate Jacobians
    J_full = Jacobian_2d(theta, links) # The full Jacobian Matrix [3X3]
    J_linear = J_full[:2, :] # Return only the linear velocity part - first 2 rows (x and y velocities) [2X3]

    # Convert direction angle to unit vector
    acceleration_direction_degrees = math.degrees(acceleration_direction)
    #print(f"acceleration_direction_degrees: {acceleration_direction_degrees}") #debugging
    theta_rad = math.radians(acceleration_direction_degrees)
    motion_direction = np.array([math.cos(theta_rad), math.sin(theta_rad)])

    try:

        # Calculate matrix inverses
        M_inv = np.linalg.inv(M) # Joint space inertia inverse

        ## LINEAR EFFECTIVE INERTIA
        # Transform joint inertia to operational space for linear motion
        Lambda_linear = np.linalg.inv(J_linear @ M_inv @ J_linear.T) # Project onto desired direction to get scalar effective inertia [2X2]
        I_eff_linear = motion_direction.T @ Lambda_linear @ motion_direction # Scalar result

        ## ANGULAR EFFECTIVE INERTIA
        # Transform joint inertia to operational space for full motion
        Lambda_full = np.linalg.inv(J_full @ M_inv @ J_full.T) # For pure rotation [3X3]
        I_eff_angular = Lambda_full[2, 2]  # Scalar result for pure end-effector rotation

        return I_eff_linear, I_eff_angular

    except np.linalg.LinAlgError:
        # Handle singular configurations (e.g., arm fully extended, joints aligned)
        # In these cases, the arm has infinite inertia in some directions
        print("Warning: Singular configuration detected")
        return 1000.0, 100.0  # Large values for singular configurations

## Joints acceleration and velocity calculations

def joint_acceleration_and_velocity_2D(theta_1, theta_2, links, links_radii, link_mass, link_inertia, gripper_param, object_param, adhesive_param,
                                       angular_energy_fraction , distance, joint_vel_limit, joint_acc_limit, S_a = 0.3, S_v = 0.2 ):

    """"""

    ## Gripper orientation and direction
    pos_start, griper_angle_start = forward_kinematics(theta_1, links)
    pos_fin, griper_angle_fin = forward_kinematics(theta_2, links)
    pos_gripper_start = pos_start[3]
    pos_gripper_fin = pos_fin[3]
    gripper_direction = pos_gripper_fin - pos_gripper_start
    gripper_direction_norm = gripper_direction / np.linalg.norm(gripper_direction)
    griper_orientation = griper_angle_start
    acceleration_direction = math.atan2(gripper_direction_norm[1],gripper_direction_norm[0])


    ## Find max linear and angular acceleration for Gripper with object based on adhesion forces
    a_linear, a_angular, F_tensile = mm.max_acceleration_2D(gripper_param,object_param, adhesive_param, griper_orientation, acceleration_direction)

    ## Find max linear and angular velocity for Gripper with object based on adhesion forces
    w_lim = mm.energy_calc(gripper_param, object_param, adhesive_param, F_tensile) # Finds available energy for Kinetic energy calculation
    I_eff_linear, I_eff_angular = calculate_effective_inertias(theta_1, links, link_mass, link_inertia, acceleration_direction) # Finds effective Inertia Matrix for linear and angular velocity
    # FIX: the previous call passed `angular_energy_fraction` POSITIONALLY into
    # the 8th slot, which is `angular_distance` -- so the requested energy split
    # (0.2) was being used as an angular distance, while the real
    # `angular_energy_fraction` silently kept its default of 0.1. Both are now
    # passed by keyword. The angular distance is the change in gripper
    # orientation over the segment.
    angular_distance = abs(griper_angle_fin - griper_angle_start)
    v_max_linear, v_max_angular, *rest = mm.max_velocity_2D(
        I_eff_linear, a_linear, acceleration_direction, distance, w_lim,
        a_angular=a_angular, I_eff_angular=I_eff_angular,
        angular_distance=angular_distance,
        angular_energy_fraction=angular_energy_fraction)  # max linear & angular velocity

    #print(f"rest: {rest}, v_max_linear: {v_max_linear}, v_max_angular: {v_max_angular}") # debugging

    ## Finds max velocity & acceleration for Joints
    if v_max_linear is None:
        # Handle error case: maybe skip, or set v_max_linear to a default, or raise exception
        raise ValueError(f"Max linear velocity calculation failed: {rest}")
    else:
        v_max_new = v_max_linear * S_v

    v_max = np.array([ # v_max as a vector - [v_x_max; v_y_max; v_angular_max]
        [v_max_new * math.cos(acceleration_direction)],
        [v_max_new * math.sin(acceleration_direction)],
        [v_max_angular * S_v]
    ])

    q_dot = rc.joint_max_velocity(theta_1, links, v_max) # Finds maximum velocity each joint is capable to move in [rad/s^2]

    for i in range(len(q_dot)): # Checks that the velocity is not higher than allowed by the robot
        if q_dot[i] < joint_vel_limit[i][1]:
            q_dot[i] = joint_vel_limit[i][1]
        elif  q_dot[i] > joint_vel_limit[i][0]:
            q_dot[i] = joint_vel_limit[i][0]

    a_max = np.array([ # a_max as a vector - [a_x_max; a_y_max; a_angular_max]
        [a_linear[0] * S_a],
        [a_linear[1] * S_a],
        [a_angular * S_a]
    ])

    #print(f"a max: {a_max}") # debugging

    q_dot_dot = rc.joint_max_acceleration(theta_1, links, q_dot, a_max) # Finds maximum acceleration each joint is capable to move in [rad/s^2]

    for i in range(len(q_dot_dot)): # Checks that the acceleration is not higher than allowed by the robot
        if q_dot_dot[i] < joint_acc_limit[i][1]:
            q_dot_dot[i] = joint_acc_limit[i][1]
        elif  q_dot_dot[i] > joint_acc_limit[i][0]:
            q_dot_dot[i] = joint_acc_limit[i][0]

    return q_dot, q_dot_dot

## Distance & Timew calculations RRT*

def wrapped_diff(a, b):
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def config_distance(config1, config2):
    diff = wrapped_diff(config1, config2)
    return np.linalg.norm(diff)

def interpolate_path(config_new, config_old, steps = 20):

    """
    Generates course - angles between start and finish configurations to check complete path

    :param config_new: The next config for the three joints (angles)
    :param config_old: The starting config for the three joints (angles)
    :param steps: Number of steps in the course, changeable
    :return: List of intermediate joint configurations
    """

    # Convert to np
    config_old = np.array(config_old)
    config_new = np.array(config_new)

    path_config = [] # Initialization of the config list

    # Finding the intermediate configurations
    for i in range(steps + 1):
            t = i / steps
            # Linear interpolation in joint space
            theta_interp = (1 - t) * config_old + t * config_new
            path_config.append(theta_interp.tolist())

    return path_config

def segment_time(theta_start, theta_fin, q_dot, q_dot_dot, q_dot_start = None):

    """"""

    ## Parameters
    #print(f"q_dot: {q_dot}, q_dot_dot: {q_dot_dot}") # debugging
    t = np.zeros_like(theta_start, dtype=float) # The segment time starts at 0
    dt = 0.01 # The time step the position change is calculated for, can be changed
    theta_start = np.array(theta_start, dtype=float)
    theta_fin = np.array(theta_fin, dtype=float)
    q_dot_max = np.array(q_dot, dtype=float).flatten()
    q_dot_dot = np.array(q_dot_dot, dtype=float).flatten()

    #print(f"q_dot: {q_dot_max}, q_dot_dot: {q_dot_dot}")  # debugging


    if q_dot_start is None:
        q_dot_start = np.zeros_like(theta_start, dtype=float)
    else:
        q_dot_start = np.array(q_dot_start, dtype=float).flatten()

    # Working copies for joint angles and velocities
    theta_current = theta_start.copy()
    q_dot_current = q_dot_start.copy()

    for i in range(len(theta_start)):
        direction = np.sign(theta_fin[i] - theta_start[i])
        if direction == 0:
            continue

        iteration_counter = 0  # Safety counter to avoid infinite loops
        while True:
            distance_remaining = direction * (theta_fin[i] - theta_current[i])
            if distance_remaining <= 0 or iteration_counter > 10000:
                print(f"distance_remaining: {distance_remaining}, distace: {theta_fin[i] - theta_start[i]}")
                break

            if abs(direction * q_dot_current[i]) < abs(direction * q_dot_max[i]):
                alpha = direction * (q_dot_dot[i])  # accelerate toward direction
            else:
                alpha = 0  # maintain current velocity

            theta_current[i] += q_dot_current[i] * dt + 0.5 * alpha * dt ** 2
            q_dot_current[i] += alpha * dt

            t[i] += dt
            iteration_counter += 1

        # Update position
        theta_current[i] += q_dot_current[i] * dt

    # Total time for arm: maximum time of all joints
    t_acc = np.max(t)
    return t_acc
