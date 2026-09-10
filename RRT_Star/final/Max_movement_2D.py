## Libraries
import math
import numpy as np

## Imported Functions
from Max_Peel_Force import max_peel_force

## Finds the max acceleration and speed allowed for the movement of the gripper

def max_acceleration_2D(gripper_param, object_param, adhesive_param, gripper_orientation, acceleration_direction):
    """
    Calculated the maximum allowed linear acceleration for the arm lifting an object in a specific direction
    """

    ## Parameters initiation
    # Adhesive parameters
    b_ad = adhesive_param['width']
    t_ad = adhesive_param['thickness']
    E_ad = adhesive_param['E']
    v_ad = adhesive_param['poisson_ratio']
    tao_y_ad = adhesive_param['shear_strength']

    # Gripper parameters
    a_gr = gripper_param['length'] # Length of contact area between gripper and object
    b_gr = gripper_param['width'] # Width of contact area between gripper and object
    theta_A = math.radians(gripper_param['peel_angle_A'])
    theta_B = math.radians(gripper_param['peel_angle_B'])

    # Scenario parameters
    h_obj = object_param['height']
    m_obj = object_param['mass']
    r_cm = object_param['r_cm']
    gamma = adhesive_param['surface_energy'] # Surface energy needed to separate the adhesive from the object
    theta_tilt = math.radians(gripper_orientation)
    #print(f"acceleration_direction: {acceleration_direction}") # debugging
    #theta_a = math.radians(acceleration_direction) # debugging
    theta_a = acceleration_direction
    r_gravity = np.array([r_cm[0]*np.cos(theta_tilt) - r_cm[1]*np.sin(theta_tilt), r_cm[0]*np.sin(theta_tilt) + r_cm[1]*np.cos(theta_tilt) ])
    g = np.array([0, -9.81]) # Gravity


    ## Known sizes calculation
    A_ad = t_ad * b_ad # Adhesive strip cross-section
    ##A_gr = a_gr * b_gr # Contact area between gripper and object
    I_ad = (b_ad * (t_ad ** 3)) / 12 # Adhesive strip second moment of inertia
    G_ad = E_ad / (2 * (1+v_ad)) # Shear modulus
    K_ad = G_ad / t_ad # Stiffness of adhesive strip
    I_obj = (m_obj *((a_gr ** 2)+(b_gr ** 2))) / 12
    # FIX: I_ad = b*t^3/12 is a SECOND MOMENT OF AREA (m^4), used for beam
    # bending; I_obj = m(a^2+b^2)/12 is a MASS MOMENT OF INERTIA (kg*m^2), used
    # for rotational dynamics. Adding them is dimensionally invalid. Only the
    # mass moment governs angular acceleration, so I_ad is dropped.
    # (Numerically I_ad ~ 9e-16 was negligible against I_obj ~ 1e-4 anyway.)
    I_total = I_obj


    ## Coefficient Calculation
    # FIX: dimensionally consistent tensile decay coefficient.
    # The previous form  k = sqrt(E_ad*I_ad/(gamma*b_gr))  evaluates to units of
    # METRES, but k is used as a decay coefficient in exp(-k*a_gr) and as 1/k^2
    # in the torque integrals, so it must have units of 1/m. With the previous
    # form k = 9.7e-6 m, giving 1/k^2 = 1.07e10 and a tensile torque of
    # -1.5e5 N*m -> angular acceleration of -1.4e9 rad/s^2.
    # This is the Kaelble decay coefficient as developed in the thesis
    # (sec 5.2.3.2):   alpha = ( E_a * b / (4 * E_s * I * a) ) ^ 1/4
    #   units: (Pa*m)/(Pa*m^4*m) = 1/m^4  ->  ^0.25 -> 1/m   (consistent)
    E_backing_k = adhesive_param.get('E_backing', E_ad)
    k = ((E_ad * b_gr) / (4.0 * E_backing_k * I_ad * t_ad)) ** 0.25
    # FIX: dimensionally consistent shear-lag decay coefficient.
    # The previous form  beta = sqrt(K_ad/(G_ad*A_ad))  evaluates to units of
    # m^-1.5, but a decay coefficient must have units of m^-1. The standard
    # shear-lag result for a bonded joint is
    #       beta = sqrt( G_a / (t_a * E_s * t_s) )
    # with G_a the adhesive shear modulus, t_a the adhesive layer thickness,
    # and E_s, t_s the adherend (backing) modulus and thickness.
    #   units: Pa / (m * Pa * m) = 1/m^2  ->  sqrt -> 1/m   (consistent)
    E_backing = adhesive_param.get('E_backing', E_ad)
    t_backing = adhesive_param.get('backing_thickness', t_ad)
    beta = math.sqrt(G_ad / (t_ad * E_backing * t_backing))


    """
    Linear acceleration
    """

    ## Forces calculation
    # Vector directions
    n_hat = np.array([math.cos(theta_tilt), math.sin(theta_tilt)])  # Normal to adhesive interface (Tensile)
    t_hat = np.array([-math.sin(theta_tilt), math.cos(theta_tilt)]) # Tangent to adhesive interface (Shear)
    a_direction = np.array([math.cos(theta_a), math.sin(theta_a)]) # Acceleration direction
    a_inertial_normal = np.dot(a_direction, n_hat)
    a_inertial_shear = np.dot(a_direction, t_hat)

    # Gravity force
    F_gravity = m_obj * g
    F_gravity_normal = np.dot(F_gravity, n_hat)
    F_gravity_shear = np.dot(F_gravity, t_hat)

    # Distance from gripper center to both edges in gripper coordinates
    edge_A_gr = np.array([-a_gr/2, 0]) # Edge A notated as negative
    edge_B_gr = np.array([a_gr / 2, 0]) # Edge B notated as positive

    # Transformation to global coordinates
    rm = np.array([[np.cos(theta_tilt), -np.sin(theta_tilt)],
                  [np.sin(theta_tilt), np.cos(theta_tilt)]]) # rotation matrix from gripper coordinates to global coordinates
    edge_A_global = np.dot(rm, edge_A_gr)
    edge_B_global = np.dot(rm, edge_B_gr)

    # Moments arms from each edge to center of mass in global coordinates
    r_A_to_cm = r_gravity - edge_A_global # gravity is from CM, already in global coordinates
    r_B_to_cm = r_gravity - edge_B_global

    ## Determining if the force at A & B is tensile or compressive
    # External forces and moments
    a_unit = 1.0 # Using unit acceleration for testing
    F_external = F_gravity + m_obj * (a_direction * a_unit) # mg + ag in global coordinates
    M_external_about_cm = np.cross(r_gravity, F_external)

    # Force distribution analysis
    # Assuming equilibrium:
    # F_A + F_B = total_normal_force
    # F_A * dist_A - F_B * dist_B = moment_from_external_forces
    total_normal_force = np.dot(F_external, n_hat)
    moment_arm_A = np.cross(r_A_to_cm, n_hat)  # Perpendicular distance from A to line of action
    moment_arm_B = np.cross(r_B_to_cm, n_hat)  # Perpendicular distance from B to line of action

    # Solve for individual forces @ A,B
    if abs(moment_arm_A - moment_arm_B) > 1e-10:
        # Calculate force distribution based on moment equilibrium
        F_A_normal = (total_normal_force * moment_arm_B + M_external_about_cm) / (moment_arm_A + moment_arm_B)
        F_B_normal = total_normal_force - F_A_normal
    else:
        # If moment arms are equal, forces split equally
        F_A_normal = total_normal_force / 2
        F_B_normal = total_normal_force / 2

    # Determine force types based on calculated normal forces
    force_type_A = 'tensile' if F_A_normal > 0 else 'compressive'
    force_type_B = 'tensile' if F_B_normal > 0 else 'compressive'

    # Calculate critical forces for each edge with proper force type consideration
    Pcr_a = max_peel_force(a_gr, b_gr, t_ad, theta_A, E_ad, gamma, force_type_A)
    Pcr_b = max_peel_force(a_gr, b_gr, t_ad, theta_B, E_ad, gamma, force_type_B)

    if Pcr_a is None or Pcr_b is None:
        return None, None  # Checking for under definition of the critical force

        # Apply modified failure criterion considering different loading types
        # The critical force is limited by whichever edge fails first
    # FIX (dimensional): the previous expressions were of the form
    #       F = (Pcr / k) * (1 - exp(-k*a_gr))
    # Pcr is a FORCE [N] and k a decay coefficient [1/m], so Pcr/k has units of
    # N*m -- not a force. This was only masked because the old `k` was itself
    # dimensionally wrong (units of metres) and small enough that
    # (1 - exp(-k*a)) ~ k*a, making F ~ Pcr*a_gr. The two errors cancelled.
    # With k corrected the cancellation disappears and F collapsed to ~1e-3 N.
    #
    # Pcr as returned by the peel models IS the critical load for the edge
    # (Kendall Eq. 3.9 for the Kaelble branch: P_cr = gamma*b/(1-sin(theta))),
    # so it is used directly. The exponential term described the stress
    # DISTRIBUTION ahead of the peel front, not a reduction in capacity, and is
    # therefore not applied as a scaling on the failure load.
    # FLAGGED FOR REVIEW -- see notes.
    if force_type_A == 'tensile' and force_type_B == 'tensile':
        # Both edges in tension - biaxial interaction criterion
        F_tensile = math.sqrt((Pcr_a ** 2 * Pcr_b ** 2) / (Pcr_a ** 2 + Pcr_b ** 2))
    elif force_type_A == 'compressive' and force_type_B == 'compressive':
        # Both edges in compression - use minimum compressive limit
        F_tensile = min(Pcr_a, Pcr_b)
    else:
        # Mixed loading - the tensile edge governs; the compressive edge supports
        F_tensile = Pcr_a if force_type_A == 'tensile' else Pcr_b

    # Shear Force calculation
    Pcr_shear = tao_y_ad * A_ad  # Critical shear force for failure

    # Calculate maximum shear force
    if beta * a_gr > 0:
        # NUMERICAL FIX: the original form
        #   (sinh(u) - sinh(-u)) / cosh(u),  u = 0.5*beta*a_gr
        # is identically 2*tanh(u), but overflows for u >~ 710 because sinh and
        # cosh each blow up before their ratio is taken. With realistic thin
        # adhesive layers (t_ad ~ 6e-5 m) beta becomes ~7e4 and u ~ 3.7e3, which
        # raised OverflowError. tanh() is the exact same value and is bounded.
        u = 0.5 * beta * a_gr
        F_shear_max = (Pcr_shear * 2.0 * math.tanh(u)) / beta
    else:
        F_shear_max = Pcr_shear

    # Remaining forces in each direction (tensile and shear)
    F_tensile_available = F_tensile - abs(F_gravity_normal) if F_gravity_normal < 0 else F_tensile + F_gravity_normal
    F_shear_available = F_shear_max - abs(F_gravity_shear)

    # Debugging
    #print(f"Debug: Force types - A: {force_type_A}, B: {force_type_B}")
    #print(f"Debug: F_A_normal: {F_A_normal:.2f} N, F_B_normal: {F_B_normal:.2f} N")
    #print(f"Debug: Pcr_a: {Pcr_a:.2f} N, Pcr_b: {Pcr_b:.2f} N")

    # Maximum acceleration from tensile constraint
    if abs(a_inertial_normal) > 1e-10:
        a_max_tensile = F_tensile_available / (m_obj * abs(a_inertial_normal))
    else:
        a_max_tensile = float('inf')

    # Maximum acceleration from shear constraint
    if abs(a_inertial_shear) > 1e-10:
        a_max_shear = F_shear_available / (m_obj * abs(a_inertial_shear))
    else:
        a_max_shear = float('inf')

    # Overall maximum acceleration
    a_max = min(a_max_tensile, a_max_shear)

    # Calculate linear acceleration vector in movement direction
    a_linear = a_max * a_direction

    #print(f"a_max: {a_max}, a_direction: {a_direction}") # debugging

    """
    Angular acceleration
    """

    ## Torque calculation
    # Torque due to tensile forces
    tao_tensile_a = (Pcr_a / (2 * (k ** 2))) * ((a_gr * k + 2) * (math.e ** (-k * a_gr)) - k * a_gr - 2) # Torque from peel force in edge A
    tao_tensile_b = Pcr_b * ((1 - (math.e ** (-k * a_gr))) / (k ** 2) - (a_gr * (1 + (math.e ** (-k * a_gr)))) / (2 * k)) # Torque from peel force in edge B
    tao_tensile = tao_tensile_a + tao_tensile_b # Total torque from forces

    # Torque due to gravity
    tao_gravity = r_gravity[0] * F_gravity[1] - r_gravity[1] * F_gravity[0]  # in 2D

    # Torque due to shear force (assuming it acts at center of object height)
    F_shear_actual = m_obj * np.dot(a_linear, t_hat)
    T_shear = F_shear_actual * (h_obj / 2)

    # Torque due to linear acceleration
    tao_a_linear = (a_linear[0] * r_gravity[1] - a_linear[1] * r_gravity[0]) * m_obj # in 2D

    # Total torque
    tao_total = tao_tensile + tao_gravity + T_shear - tao_a_linear

    # Max angular acceleration
    a_angular = tao_total / I_total

    #print(f"a_linear: {a_linear}, a_angular: {a_angular}")  # debugging

    return a_linear, a_angular, F_tensile

def energy_calc(gripper_param, object_param, adhesive_param, F_tensile):
    """"""

    gripper_length = gripper_param['length']
    gripper_width = gripper_param['width']
    object_length = object_param['length']
    object_width = object_param['width']
    adhesive_length = adhesive_param['length']
    adhesive_width = adhesive_param['width']
    adhesive_thickness = adhesive_param['thickness']
    surface_energy = adhesive_param['surface_energy']
    adhesive_youngs_modulus = adhesive_param['E']
    adhesive_max_strain = adhesive_param['strain']

    ## Parameters

    l_cont = min(gripper_length, object_length, adhesive_length) # Finds minimum contact length
    w_cont = min(gripper_width ,object_width ,adhesive_width) # Finds minimum contact width
    a_cont = l_cont * w_cont # Contact area between adhesive and object
    v_adh = adhesive_length * adhesive_width * adhesive_thickness # Adhesive Volume

    ## Method 1 - Surface Energy calculation: gamma*A_contact
    w_1 = surface_energy * a_cont

    # Method 2 - Average Peel Force Work: 0.5 * F_peel * Peel_distance
    w_2 = 0.5 * F_tensile * l_cont

    # Method 3 - Elastic Strain Energy: REMOVED.
    # w_3 = 0.5*E*V*strain^2 is structurally ~1e-10 J for any physically
    # realistic small strain, i.e. ~10 orders of magnitude below w_1 and w_2
    # (~0.1-1 J). It also measures a different quantity: recoverable elastic
    # STORAGE, rather than the energy required to DEBOND. Including it in the
    # min() therefore always selected it and rendered every motion infeasible.

    # Energy limit = the smaller of the two debonding-energy estimates.
    w_lim = min(w_1, w_2)

    #print(f"w_1: {w_1}, w_2: {w_2}, w_lim: {w_lim}, F: {F_tensile}")  # debugging

    return w_lim

def max_velocity_2D(m_eff_linear, a_linear, acceleration_direction, distance_moved, w_lim,
                    a_angular=None, I_eff_angular=None, angular_distance=0,
                    v_initial=0, omega_initial=0, angular_energy_fraction=0.1):
    """

    :param obj_param: The Parameters of the object
    :param a_linear: Maximum linear acceleration (m/s²)
    :param acceleration_direction: The gripper's direction of movement
    :param distance_moved: The distance between the two checked nodes
    :param w_lim: The available energy for velocity calc
    :param a_angular: Maximum angular acceleration (rad/s²)
    :param I_eff_angular: Moment of inertia about rotation axis (kg⋅m²)
    :param angular_distance: Angular distance to rotate (radians)
    :param v_initial: initial linear velocity (m/s)
    :param omega_initial: Initial angular velocity (rad/s)
    :param angular_energy_fraction: Fraction of energy allocated to angular motion

    :return: A vector for v_max [v_x_max, v_y_max] [m/s] & the size v_angular_max [rad/s]
    """

    ## Parameters
    g = 9.81  # Gravity acceleration
    a_mag = np.linalg.norm(a_linear)  # Finds the size of the acceleration in the acceleration direction
    #print(f"acceleration_direction: {acceleration_direction}, a_mag: {a_mag}") # debugging
    theta_a = acceleration_direction  # Theta of the direction of acceleration
    # FIX: this term is the EFFECTIVE LINEAR MASS. It is used both as the mass
    # in the gravity work  w_g = m*g*sin(theta)*d  and in the kinetic-energy
    # inversion  v_f^2 = 2*w_net/m.  It was previously hardcoded to 1 kg, which
    # overstated the required energy ~10x for the 0.1 kg object.
    # NOTE: the first positional parameter was misleadingly named `obj_param`
    # but every caller passes `I_eff_linear` (a scalar effective mass), so it is
    # now named accordingly and used directly.
    m_eff = float(m_eff_linear)
    d = distance_moved

    ## LINEAR VELOCITY CALCULATION

    ## Method 1 - Energy calculation:

    # Split energy between linear and angular motion
    w_linear = w_lim * (1 - angular_energy_fraction)
    w_angular = w_lim * angular_energy_fraction

    # Correct energy to include the part lost for gravity while moving
    w_g = m_eff * g * math.sin(theta_a) * d  # Work invested into gravity [J]
    print(f"w lim: {w_lim}, wg: {w_g}, d: {d}, I: {m_eff}")
    w_net_linear = w_linear - w_g  # Remaining energy for linear motion [J]

    print(f"w net: {w_net_linear}")

    if w_net_linear <= 0:
        return None, None, "Insufficient linear energy to overcome gravity"

    # Find Final Linear Velocity
    v_f_sqrt_e = ((2 * w_net_linear) / m_eff) + v_initial ** 2

    if v_f_sqrt_e < 0:
        return None, None, "Cannot achieve positive final velocity"

    v_f_e = math.sqrt(v_f_sqrt_e)  # Max allowed velocity from energy calc

    # Limitation Check
    a_rec_e = (v_f_e ** 2 - v_initial ** 2) / (2 * d) if d > 0 else 0

    if a_rec_e > a_mag:  # The needed acceleration is larger than the allowed acceleration
        v_f_e = math.sqrt(2 * a_mag * d + (v_initial ** 2))

    ## Method 2 - Force Analysis (your existing code)

    # Force components calculation
    f_max = m_eff * a_mag  # Maximum Force from the Adhesive
    f_g = m_eff * g * math.sin(theta_a)  # Gravity Force component, opposing motion
    f_net = f_max - f_g  # Remaining Available Force

    if f_net <= 0:  # Gravity Force too large for movement
        return None, None, "Cannot overcome gravity with available force"

    # Final velocity from kinematics: v_f**2 = v_initial**2 + 2 * a * d
    a_net = f_net / m_eff  # The actual acceleration possible for movement
    v_f_sqrt_f = (v_initial ** 2) + (2 * a_net * d)

    if v_f_sqrt_f < 0:
        return None, None, "Cannot achieve positive final velocity"

    v_f_f = math.sqrt(v_f_sqrt_f)  # Final velocity allowed by Force Analysis

    # Choose more restrictive linear velocity
    if v_f_f > v_f_e:
        v_max_linear = v_f_e
    else:
        v_max_linear = v_f_f

    ## ANGULAR VELOCITY CALCULATION

    if a_angular is None or I_eff_angular is None:
        # No angular motion possible
        v_max_angular = omega_initial
    else:
        ## Angular Method 1 - Energy calculation

        # Available rotational energy
        if w_angular > 0:
            # Rotational kinetic energy: KE = 0.5 * I * ω²
            omega_f_sqrt_e = ((2 * w_angular) / I_eff_angular) + omega_initial ** 2

            if omega_f_sqrt_e >= 0:
                omega_f_e = math.sqrt(omega_f_sqrt_e)
            else:
                omega_f_e = abs(omega_initial)
        else:
            omega_f_e = abs(omega_initial)

        # Check if energy-based angular velocity requires too much acceleration
        if angular_distance > 0:
            alpha_required_e = (omega_f_e ** 2 - omega_initial ** 2) / (2 * angular_distance)
            if alpha_required_e > abs(a_angular):
                omega_f_e = math.sqrt(2 * abs(a_angular) * angular_distance + omega_initial ** 2)

        ## Angular Method 2 - Torque/Acceleration Analysis

        # Maximum torque available
        tau_max = abs(a_angular) * I_eff_angular

        # Angular velocity from kinematics: ω² = ω₀² + 2αθ
        if angular_distance > 0:
            omega_f_sqrt_f = omega_initial ** 2 + 2 * abs(a_angular) * angular_distance
            if omega_f_sqrt_f >= 0:
                omega_f_f = math.sqrt(omega_f_sqrt_f)
            else:
                omega_f_f = abs(omega_initial)
        else:
            # If no specific angular distance, estimate based on time
            # Assume angular motion occurs over same time as linear motion
            if v_max_linear > 0 and d > 0:
                motion_time = d / (v_max_linear / 2)  # Rough time estimate
                omega_f_f = abs(omega_initial + a_angular * motion_time)
            else:
                omega_f_f = abs(omega_initial)

        # Choose more restrictive angular velocity
        v_max_angular = min(omega_f_e, omega_f_f)

        # Apply reasonable physical limits
        max_reasonable_omega = 10.0  # rad/s - adjust based on your system
        v_max_angular = min(v_max_angular, max_reasonable_omega)

    return v_max_linear, v_max_angular