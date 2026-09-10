## Libraries
import numpy as np
from typing import Dict, List, Union, Any

## Imported Functions
import Kinematics_and_Dynamics as kd


## Self collision check-VX, Obstacle check-VX, Geometry check-VX, Angle check-VX,  Singularity detection, Acceleration check, Velocity check, Robot torque check

## Obstacle and Self collision check
def capsule_to_circle_distance(p1, p2, link_radius, circle_center, circle_radius):
    """
    Calculate minimum distance between a link and a circle type obstacle
    Link is defined as a capsule (line with thickness)

    Args:
        p1: Startpoint of the link centerline
        p2: Endpoint of the link centerline
        link_radius: Radius of the capsule
        circle_center: Center of the circle obstacle
        circle_radius: Radius of the circle obstacle

    Returns:
        Minimum distance between capsule surface and circle surface
    """

    # Distance from line segment to circle center
    line_vec = np.array(p2 - p1)
    line_length = np.linalg.norm(line_vec)

    if line_length == 0:
        # Link is just a circle
        center_distance = np.linalg.norm(p1 - circle_center)
        return max(0, center_distance - link_radius - circle_radius)

    # Normalized line direction
    line_dir = line_vec / line_length

    # Vector from p1 to circle center
    to_center = circle_center - p1

    # Project circle center onto line segment
    projection_length = np.dot(to_center, line_dir)
    projection_length = np.clip(projection_length, 0, line_length)

    # Closest point on line segment centerline
    closest_point = p1 + projection_length * line_dir

    # Distance from the closest point to circle center
    center_distance = np.linalg.norm(circle_center - closest_point)

    # Subtract both radii to get surface-to-surface distance
    return max(0, center_distance - link_radius - circle_radius)


def capsule_to_rectangle_distance(p1, p2, link_radius, rect_bottom_left, rect_width, rect_height):
    """
    Minimum surface distance from a capsule (segment + radius) to an axis-aligned
    rectangle.

    PERFORMANCE FIX: the previous version looped over 201 sampled points in pure
    Python, calling max() six times per point. Profiling showed 18,360 calls
    generating 7.5 million max() evaluations and dominating runtime once the
    capsule-capsule bottleneck was removed.

    The sampling is retained (a closed-form segment-to-AABB distance is more
    intricate than segment-to-segment) but is now fully vectorised in NumPy, and
    the sample count is unchanged so the numerical result is identical.
    """
    rect_bottom_left = np.asarray(rect_bottom_left, dtype=float)
    rect_top_right = rect_bottom_left + np.array([rect_width, rect_height], dtype=float)
    p1 = np.asarray(p1, dtype=float); p2 = np.asarray(p2, dtype=float)

    num_points = 200
    t = np.linspace(0.0, 1.0, num_points + 1).reshape(-1, 1)
    pts = p1 + t * (p2 - p1)                      # (N,2) sampled points
    pts = np.vstack([pts, p1.reshape(1, -1), p2.reshape(1, -1)])   # include endpoints

    dx = np.maximum.reduce([rect_bottom_left[0] - pts[:, 0],
                            np.zeros(len(pts)),
                            pts[:, 0] - rect_top_right[0]])
    dy = np.maximum.reduce([rect_bottom_left[1] - pts[:, 1],
                            np.zeros(len(pts)),
                            pts[:, 1] - rect_top_right[1]])
    return float(np.min(np.sqrt(dx * dx + dy * dy)) - link_radius)


def capsule_to_capsule_distance(p1_start, p1_end, r1, p2_start, p2_end, r2):
    """
    Minimum surface-to-surface distance between two capsules.

    PERFORMANCE FIX: the previous implementation sampled a 101x101 grid of point
    pairs along the two segments (10,201 np.linalg.norm calls per invocation).
    Profiling a 15-iteration planning run showed 1,530 calls to this function
    generating 15.6 MILLION norm evaluations and accounting for ~96% of total
    runtime, which is why planning never completed.

    This is the standard closed-form segment-to-segment distance (Ericson,
    "Real-Time Collision Detection", sec. 5.1.9). It is O(1), and being exact it
    is also MORE accurate than the sampled version, which could only ever
    approximate the true minimum to the resolution of its grid.
    """
    d1 = np.asarray(p1_end, dtype=float) - np.asarray(p1_start, dtype=float)
    d2 = np.asarray(p2_end, dtype=float) - np.asarray(p2_start, dtype=float)
    r = np.asarray(p1_start, dtype=float) - np.asarray(p2_start, dtype=float)

    a = float(np.dot(d1, d1))       # squared length of segment 1
    e = float(np.dot(d2, d2))       # squared length of segment 2
    f = float(np.dot(d2, r))
    EPS = 1e-12

    if a <= EPS and e <= EPS:                     # both degenerate to points
        return float(np.linalg.norm(r)) - r1 - r2
    if a <= EPS:                                  # segment 1 is a point
        s = 0.0
        t = min(max(f / e, 0.0), 1.0)
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:                              # segment 2 is a point
            t = 0.0
            s = min(max(-c / a, 0.0), 1.0)
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = min(max((b * f - c * e) / denom, 0.0), 1.0) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t = 1.0
                s = min(max((b - c) / a, 0.0), 1.0)

    closest1 = np.asarray(p1_start, dtype=float) + d1 * s
    closest2 = np.asarray(p2_start, dtype=float) + d2 * t
    return float(np.linalg.norm(closest1 - closest2)) - r1 - r2


def check_collision_links(angles, links, link_radii, obstacles, safety_margin=0.0):
    """
    Check collision for robotic arm with thick (circular cross-section) links.
    Links defined as capsule
    Checks self collision (between links) and with obstacles

    Args:
        angles: List of 3 joint angles in radians
        links: List of 4 link lengths [base_height, link1, link2, link3]
        link_radii: List of link radii [r1, r2, r3, r4] or single radius for all links
        obstacles: List of obstacle dictionaries with optional 'radius' for thick obstacles
        safety_margin: Additional safety distance, specify by user

    Returns:
        collision_info: Dictionary with collision details
    """

    # Handle link_radii input
    if isinstance(link_radii, (int, float)):
        link_radii = [link_radii] * 4  # Same radius for all links
    elif len(link_radii) != 4:
        raise ValueError("link_radii must be a single value or list of 4 values")

    # Extract link segments from your FK function
    joint_positions, link_segments = kd.extract_link_segments_from_fk(angles, links)

    if joint_positions is None:
        return {'collision': True, 'error': 'FK function failed'}

    collision_info: Dict[str, Union[bool, float, List[Union[Dict[str, Union[int, float, str]], Any]]]] = {
        'collision': False,  # Initialise collision as no collision detected
        'min_distance': float('inf'),  # Initialise min distance between link and objects
        'colliding_links': [],  # List of links that collide with an object
        'distances': [],  # Array of distances between Links and Objects (now stores dicts)
        'joint_positions': joint_positions,  # Position of base and 3 Joints from FK
        'link_segments': link_segments,  # Vectors of the three links in the world
        'link_radii': link_radii  # The radius of the 3 links (2 links and the end effector)
    }

    # Check each link against each obstacle
    for link_idx, (start_pos, end_pos) in enumerate(link_segments):
        capsule_radius = link_radii[link_idx]

        # Different check for the 3 types of obstacles
        for obs_idx, obstacle in enumerate(obstacles):

            if obstacle['type'] == 'circle':
                distance = capsule_to_circle_distance(
                    start_pos, end_pos, capsule_radius,
                    np.array(obstacle['center']),
                    obstacle['radius']
                )
            elif obstacle['type'] == 'rectangle':
                distance = capsule_to_rectangle_distance(
                    start_pos, end_pos, capsule_radius,
                    np.array(obstacle['bottom_left']),
                    obstacle['width'],
                    obstacle['height']
                )
            elif obstacle['type'] == 'capsule':
                # Thick line obstacle
                distance = capsule_to_capsule_distance(
                    start_pos, end_pos, capsule_radius,
                    np.array(obstacle['start']), np.array(obstacle['end']),
                    obstacle['radius']
                )
            else:
                raise ValueError(f"Unknown obstacle type: {obstacle['type']}")

            collision_info['distances'].append({
                'link': link_idx,
                'obstacle': obs_idx,
                'distance': distance,
                'link_radius': capsule_radius
            })

            collision_info['min_distance'] = min(collision_info['min_distance'], distance)

            if distance <= safety_margin:
                collision_info['collision'] = True
                if link_idx not in collision_info['colliding_links']:
                    collision_info['colliding_links'].append(link_idx)

    # Check self-collision between non-adjacent links
    for i in range(len(link_segments)):
        for j in range(i + 2, len(link_segments)):  # Skip adjacent links
            start1, end1 = link_segments[i]
            start2, end2 = link_segments[j]
            r1, r2 = link_radii[i], link_radii[j]

            distance = capsule_to_capsule_distance(start1, end1, r1, start2, end2, r2)

            collision_info['distances'].append({
                'link': i,
                'obstacle': f'self_link_{j}',
                'distance': distance,
                'link_radius': r1,
                'type': 'self_collision'
            })

            collision_info['min_distance'] = min(collision_info['min_distance'], distance)

            if distance <= safety_margin:
                collision_info['collision'] = True
                if i not in collision_info['colliding_links']:
                    collision_info['colliding_links'].append(i)
                if j not in collision_info['colliding_links']:
                    collision_info['colliding_links'].append(j)

    # FIXED: Ground collision check - moved outside the self-collision loop
    pos, pos_angle = kd.forward_kinematics(angles, links)
    for r in range(1, len(pos)):
        if pos[r][1] <= 0:
            collision_info['collision'] = True

    return collision_info


## Robot arm limitation check
def geometry_check(point, cylinder_center, cylinder_height, cylinder_radius):
    """

    :param point: Position of the Joints (x,y)
    :param cylinder_center: The middle of the link's volume (x,y)
    :param cylinder_height: The link's length
    :param cylinder_radius: THe link's radius
    :return:
    """

    # Checks that the point is within the cylinder's height
    if abs(point[1] - cylinder_center[1]) > cylinder_height:
        return False  # Collision detected

    # Checks that the point is within the cylinder's radius
    if abs(point[0] - cylinder_center[0]) <= cylinder_radius:
        return False  # Collision detected

    return True


def angle_check(angles, joint_limits):
    # Gets all the Min and Max restrictions from the one array into two arrays
    min_angles = joint_limits[:, 0]  # FIXED: Changed from index 1 to 0 for min values
    max_angles = joint_limits[:, 1]  # FIXED: Changed from index 2 to 1 for max values

    angle_violation_info = {
        'violation': False,  # Initialise violation as no violation detected
        'angles': angles,  # FIXED: Store actual angles array instead of float('inf')
        'violating_joints': [],  # FIXED: List of joint indices that violated limits
    }

    # Checks angle limitations
    for i in range(len(angles)):
        if angles[i] < min_angles[i] or angles[i] > max_angles[i]:
            angle_violation_info['violation'] = True  # Impossible angle for Joint
            angle_violation_info['violating_joints'].append(i)

    return angle_violation_info


def singular_check(angles, links, tol=1e-6):
    """

    :param angles: Angeles of the 3 Joints
    :param links: Length of the links - link[0] is the distance from the ground
    :param tol: The tolerance for detecting singularity
    :return:
    """

    J = kd.Jacobian_2d(angles, links)
    det_J = np.linalg.det(J)
    if np.abs(det_J) < tol:
        return False  # Singularity found

    return True  # No Singularity detected


def limitation_check(angles, links, link_radius, joint_limits):
    """
    Checks if the new position is possible with the robotics arm limitations
    :param angles: Angles of the Joints after movement, next position in the RRT*
    :param links: The length of each Link.
    :param link_radius: The thickness of each link and the end effector
    :param joint_limits: Min & Max angle possible by each Joint
    :return: angle_check, singularity_check, geometric_check
    """

    # Extract link segments from FK
    joint_positions, link_segments = kd.extract_link_segments_from_fk(angles, links)

    link_center = []
    geo_check = []

    for i in range(len(link_segments)):
        start = np.array(link_segments[i][0])
        end = np.array(link_segments[i][1])
        center = (start + end) / 2
        link_center.append(center)

    for i in range(1, len(joint_positions)):  # skip base at index 0
        point = np.array(joint_positions[i])
        center = link_center[i - 1]  # i-1 because link_center[0] is for joint 1
        height = links[i]
        radius = link_radius[i - 1]
        geo_check.append(geometry_check(point, center, height, radius))

    ang_check = angle_check(angles, joint_limits)
    sing_check = singular_check(angles, links, tol=1e-6)

    return ang_check, sing_check, geo_check


## Acceleration and Velocity check

def joint_max_velocity(theta, links, V_max):
    """
    Finds the velocity of each Joint for the max velocity allowed for the gripper

    :param theta: Angles of all 3 Joints [Joint 1, Joint 2, Joint 3]
    :param links: Length of all 4 Links [ Base height, Link 1, Link 2, Link 3]
    :param V_max: Maximum allowed velocity including a safety factor [Vx, Vy, Vt]

    :return: The velocity for each Joint [3X1]
    """

    J = kd.Jacobian_2d(theta, links)
    J_plus = np.linalg.pinv(J)
    print(f"J_plus: {J_plus}, V_max: {V_max}")
    q_dot = np.matmul(J_plus, V_max)

    return q_dot


def joint_max_acceleration(theta, links, q_dot, A_max):
    """
    Finds the acceleration of each Joint for the max acceleration allowed for the gripper

    :param theta: Angles of all 3 Joints [Joint 1, Joint 2, Joint 3]
    :param links: Length of all 4 Links [ Base height, Link 1, Link 2, Link 3]
    :param q_dot: The velocity for each Joint [3X1]
    :param A_max: Maximum allowed acceleration including a safety factor [ax, ay, at]

    :return: The acceleration for each Joint [3X1]
    """

    ## Jacobians
    J = kd.Jacobian_2d(theta, links)
    J_plus = np.linalg.pinv(J)
    J_dot = kd.Jacobian_dot_2d(theta, q_dot, links)

    ## Calculation of q_dot_dot - q_dot_dot = J_plus * (aee - J_dot * q_dot)
    Jq_dot = np.matmul(J_dot, q_dot)  # The part of the velocity of the Joints
    aee_dif = A_max - Jq_dot  # The part within the parentheses

    q_dot_dot = np.matmul(J_plus, aee_dif)

    return q_dot_dot


## Path Collision Checks

def check_path_collision(path, links, link_radius, obstacles, safety_margin=0.0):
    """

    :param path: The intermediate path for the movement
    :param links: The length of the links
    :param link_radius: The radius of the links
    :param obstacles: List of obstacle dictionaries with optional 'radius' for thick obstacles
    :param safety_margin: Additional safety distance, specify by user
    :return: Dictionary with path collision results
    """

    path_collision_info = {
        'collision': False,
        'min_distance_along_path': float('inf'),
        'collision_configurations': [],
        'collision_steps': [],
        'all_config_results': [],
        'first_collision_step': None
    }

    # Check each intermediate configuration
    for step_idx, config in enumerate(path):
        config_result = check_collision_links(
            config, links, link_radius, obstacles, safety_margin
        )

        path_collision_info['all_config_results'].append({
            'step': step_idx,
            'configuration': config,
            'result': config_result
        })

        # Checks for collisions
        if config_result['collision']:
            path_collision_info['collision'] = True
            path_collision_info['collision_configurations'].append(config)
            path_collision_info['collision_steps'].append(step_idx)

            # Record first collision for early termination if needed
            if path_collision_info['first_collision_step'] is None:
                path_collision_info['first_collision_step'] = step_idx

        # Track minimum distance along the entire path
        if config_result['min_distance'] < path_collision_info['min_distance_along_path']:
            path_collision_info['min_distance_along_path'] = config_result['min_distance']

    return path_collision_info


def check_joint_limits_along_path(path, joint_limits):
    """
    Check if path violates joint limits

    :param path: The intermediate path for the movement
    :param joint_limits: The Min and Max angles each Joint is capable of - shape (n_joints, 2) where [:, 0] is min and [:, 1] is max
    :return: Dictionary with violation information
    """

    # Ensure joint_limits has correct format
    if len(joint_limits.shape) == 1:
        raise ValueError("joint_limits should be 2D array with shape (n_joints, 2)")

    min_limits = joint_limits[:, 0]
    #print('min_limits')
    #print(min_limits) #DBug
    max_limits = joint_limits[:, 1]
    #print('max_limits')
    #print(max_limits) #DBug

    for step_idx, config in enumerate(path):
        #print ('config')
        #print (config) #DBug
        for joint_idx, angle in enumerate(config):
            if angle < min_limits[joint_idx] or angle > max_limits[joint_idx]:
                return {
                    'violation': True,
                    'joint': joint_idx,
                    'angle': angle,
                    'limits': (min_limits[joint_idx], max_limits[joint_idx]),
                    'config': config,
                    'step': step_idx
                }

    return {'violation': False}

def check_singularity_along_path(path , links, tol = 1e-6):

    """
    Checks for singularities along the path

    :param path: The intermediate path for the movement
    :param links: The length of the links
    :param tol: The tolerance for singularity to be decided
    :return: The config that gives a singular point if there is one
    """

    for step_idx, config in enumerate(path):
        J = kd.Jacobian_2d(config, links)
        det_J = np.linalg.det(J)
        diff = np.abs(det_J) - tol
        if diff < 0:
            return {
                'singularity' : True,
                'config' : config,
                'step' : step_idx
            }

    return {'singularity' : False}




