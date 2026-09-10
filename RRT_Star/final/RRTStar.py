## Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import reconstruct_path
from scipy.spatial.distance import euclidean
import random
import math

## Imported Functions
import Restriction_Checks as rc
import Kinematics_and_Dynamics as kd
import Visualization as vs

class RRTStar:

    def __init__(self, start_config, goal_config, # Start and Finish Points
               joint_limits, joint_vel_limit, joint_acc_limit, link_lengths, link_radii, safety_margin, obstacles, mass, inertia, S_a, S_v, # Limits of the robotic arms
               gripper_param, object_param, adhesive_param,  angular_energy_friction, # Parameters for acceleration calc
               step_size=0.01, goal_tolerance=0.05, max_iterations=5000, rewire_radius=0.5, w_time=1.0, w_dist=0.5, w_advance=0.7): # Parameters for the RRT*
        """


                Initialize RRT* in 2D

        :param start_config: Starting Joints angle in radians
        :param goal_config: Goal Joint angle in radians
        :param joint_limits: A list of Min and Max angle for each Joint
        :param joint_vel_limit: A list of max and min velocities for each Joint
        :param joint_acc_limit: A list of max and min acceleration for each Joint
        :param link_lengths: A list of the Length of the links
        :param link_radii: A list of the radius (thickness) of the links
        :param safety_margin: A parameter decided by the user
        :param obstacles : An Array of all the known obstacles including type and parameters
        :param S_a: Safety factor for max acceleration
        :param S_v: Safety factor for max velocity
        :param mass: The mass matrix of the robotic arm
        :param inertia: The Inertia Matrix for the robotic arm
        :param gripper_param: Geometrical parameters of the gripper
        :param object_param: Geometrical and physical parameters of the lifted object
        :param adhesive_param: Geometrical and physical parameter of the adhesive on the gripper
        :param angular_energy_friction: The amount of energy devoted to the angular kinematic part
        :param step_size: Max step size in configuration space
        :param goal_tolerance: Distance threshold for goal reached
        :param max_iterations: Max number of allowed iterations beefore failure
        :param rewire_radius: Radius for rewiring in RRT*
        :param w_time: cost weight for the time a segment took
        :param w_dist: cost weight for a segment's length
        :param w_advance: cost of not advancing in the right direction
        """

        self.start = np.array(start_config)
        self.goal = np.array(goal_config)
        self.joint_limits = np.array(joint_limits) # np.array added 03.09
        self.joint_vel_limit = joint_vel_limit
        self.joint_acc_limit = joint_acc_limit
        self.link_lengths = link_lengths
        self.link_radii = link_radii
        self.safety_margin = safety_margin
        self.obstacles = obstacles
        self.S_a = S_a
        self.S_v = S_v
        self.mass = mass
        self.inertia = inertia
        self.gripper_param = gripper_param
        self.object_param = object_param
        self.adhesive_param = adhesive_param
        self.angular_energy_friction = angular_energy_friction
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations
        self.rewire_radius = rewire_radius
        self.w_time = w_time
        self.w_dist = w_dist
        self.w_advance = w_advance

        # Tree structure: each node contains [config, parent_index, cost_from_start]
        self.tree = []
        self.tree.append([self.start.copy(), -1, 0.0])  # Root node

        # For visualization
        self.path = None

    def config_distance(self, config1, config2):
        """Calculate distance between two configurations in joint space"""

        distance = kd.config_distance(config1, config2)
        return distance

    def sample_random_config(self):
        """Sample a random valid configuration within joint limits"""
        config = []
        for i in range(len(self.joint_limits)):
            min_angle = self.joint_limits[i][0]
            max_angle = self.joint_limits[i][1]
            val = random.uniform(min_angle, max_angle)
            config.append(val)
        return np.array(config)

    def find_nearest_node(self, target_config):
        """Find the nearest node in the tree to target configuration"""
        min_dist = float('inf')
        nearest_idx = 0

        for i, (config, _, _) in enumerate(self.tree):
            dist = self.config_distance(config, target_config)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx, min_dist

    def steer(self, from_config, to_config):
        """
        Steer from one configuration towards another with step size limit
        Returns new configuration and the distance moved
        """
        direction = to_config - from_config
        print(f"Steer from {from_config} towards {to_config}. Direction: {direction}")
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return to_config, distance
        else:
            # Normalize and scale by step size
            unit_direction = direction / distance
            new_config = from_config + unit_direction * self.step_size
            return new_config, self.step_size

    def rrt_segment_cost(self, config1, config2):

        """"""
        distance = kd.config_distance(config1, config2)
        q_dot, q_dot_dot = kd.joint_acceleration_and_velocity_2D(config1, config2, self.link_lengths, self.link_radii, self.mass, self.inertia,
                                                                 self.gripper_param, self.object_param, self.adhesive_param,
                                                                 self.angular_energy_friction, distance, self.joint_vel_limit, self.joint_acc_limit)
        time = kd.segment_time(config1, config2, q_dot, q_dot_dot)
        return time

    def calculate_segment_cost(self, config1, config2, start, goal):
        """
        Calculate comprehensive cost between two configurations including time
        """
        try:
            # Distance component
            distance = self.config_distance(config1, config2)

            # Time component (this includes dynamics, acceleration, velocity constraints)
            time_cost = self.rrt_segment_cost(config1, config2)

            # Advance component
            start_goal = np.sum(goal - start)
            current_goal = np.sum(config1 - start)

            # Check for invalid values
            if np.isnan(distance) or np.isnan(time_cost) or np.isinf(distance) or np.isinf(time_cost):
                return float('inf'), float('inf'), distance

            # Combined cost with weights
            total_cost = self.w_time * time_cost + self.w_dist * distance + self.w_advance * (
                        current_goal / start_goal) ** 2

            return total_cost, time_cost, distance

        except Exception as e:
            print(f"Error in calculate_segment_cost: {e}")
            return float('inf'), float('inf'), 0.0

    def get_path_time_cost(self, node_idx):
        """Get the total time from start to given node"""
        if node_idx == 0:  # Start node
            return 0.0

        try:
            # Traverse back to get total time
            total_time = 0.0
            current_idx = node_idx

            while current_idx != -1 and self.tree[current_idx][1] != -1:  # While not at root
                parent_idx = self.tree[current_idx][1]
                current_config = self.tree[current_idx][0]
                parent_config = self.tree[parent_idx][0]

                # Get time for this segment
                segment_time = self.rrt_segment_cost(parent_config, current_config)
                if np.isnan(segment_time) or np.isinf(segment_time):
                    return float('inf')

                total_time += segment_time
                current_idx = parent_idx

            return total_time
        except Exception as e:
            print(f"Error calculating path time cost: {e}")
            return float('inf')

    def is_collision_free(self, config):
        """
        Checks if the configuration is collision free for a single config
        """

        # Gets collision results and limitation results from the functions in Restriction_Check
        collision_result = rc.check_collision_links(config, self.link_lengths, self.link_radii, self.obstacles, self.safety_margin)

        if collision_result['collision']:
            return False

        return True

    def are_limitations_answered(self, config):
        """
        Checks if the configuration is within arm limitations
        """
        # Gets checking Data from Functions
        angle_check, singularity_check, geometric_check = rc.limitation_check(config, self.link_lengths, self.link_radii, self.joint_limits)

        # Case 1 - Unreachable angle
        if angle_check['violation']:
            return False

        # Case 2 - Singular point
        if  not singularity_check:
            return False

        # Case 3 - Location out of bounds
        if not all(geometric_check):
            return False

        return True

    def is_path_valid(self, config1, config2, steps = 100):

        """

        :param config1: Starting config of the arm
        :param config2: End config of the arm
        :param steps: Number of steps along the path
        :return: Is the path valid or not
        """

        # Initiating the validation dictionary
        validation_result = {
            'valid': True,
            'collision_free': True,
            'joint_limits_ok': True,
            'collision_info': None,
            'joint_limit_info': None
        }

        path_config = kd.interpolate_path(config1, config2, steps)

        # Case 1 - Collisions along the path
        collision_result = rc.check_path_collision(path_config, self.link_lengths, self.link_radii, self.obstacles, self.safety_margin)

        if collision_result['collision']:
            validation_result['valid'] = False
            validation_result['collision_free'] = False
            validation_result['collision_info'] = collision_result

        # Case 2 - Unreachable angle
        joint_limit_result = rc.check_joint_limits_along_path(path_config, self.joint_limits)

        if joint_limit_result['violation']:
            validation_result['valid'] = False
            validation_result['joint_limits_ok'] = False
            validation_result['joint_limit_info'] = joint_limit_result

        # Case 3 - Singularity check
        sing_check = rc.check_singularity_along_path(path_config, self.link_lengths)

        if sing_check['singularity']:
            validation_result['valid'] = False
            validation_result['joint_limits_ok'] = False
            validation_result['joint_limit_info'] = joint_limit_result

        return validation_result['valid']

    def find_near_nodes(self, config, radius):
            """Find all nodes within given radius of the configuration"""
            near_indices = []
            for i, (node_config, _, _) in enumerate(self.tree):
                if self.config_distance(node_config, config) <= radius:
                    near_indices.append(i)
            return near_indices

    def get_path_cost(self, node_idx):
        """Get the cost from start to given node"""
        return self.tree[node_idx][2]

    def rewire_tree(self, new_node_idx, near_indices):
        """
        Enhanced rewire with time-aware cost calculations
        """
        try:
            new_config = self.tree[new_node_idx][0]
            new_cost = self.tree[new_node_idx][2]

            for near_idx in near_indices:
                if near_idx == new_node_idx:
                    continue

                near_config = self.tree[near_idx][0]
                near_cost = self.tree[near_idx][2]

                # Calculate cost if we route through new node (time-aware)
                segment_cost, segment_time, segment_dist = self.calculate_segment_cost(new_config, near_config, self.start, self.goal)

                if np.isinf(segment_cost):
                    continue  # Skip if cost calculation failed

                potential_cost = new_cost + segment_cost

                # If this path is better and valid, rewire
                if potential_cost < near_cost and self.is_path_valid(new_config, near_config):
                    # Update parent and cost
                    self.tree[near_idx][1] = new_node_idx  # New parent
                    self.tree[near_idx][2] = potential_cost  # New cost

                    # Recursively update costs of descendants
                    self.update_descendants_cost(near_idx)

        except Exception as e:
            print(f"Error in rewire_tree: {e}")

    def update_descendants_cost(self, node_idx):
        """Recursively update costs of all descendants after rewiring"""
        try:
            node_config = self.tree[node_idx][0]
            node_cost = self.tree[node_idx][2]

            # Find all children
            for i, (config, parent_idx, cost) in enumerate(self.tree):
                if parent_idx == node_idx:
                    # Calculate new cost using time-aware calculation
                    segment_cost, _, _ = self.calculate_segment_cost(node_config, config, self.start, self.goal)

                    if np.isinf(segment_cost):
                        continue  # Skip if cost calculation failed

                    new_cost = node_cost + segment_cost
                    self.tree[i][2] = new_cost

                    # Recursively update grandchildren
                    self.update_descendants_cost(i)
        except Exception as e:
            print(f"Error updating descendants cost: {e}")

    def is_goal_reached(self, config):
        """
        Check if configuration is close enough to goal using both
        Cartesian end-effector position and joint-space distance.
        """

        # --- Cartesian check ---
        config_pos, _ = kd.forward_kinematics(config, self.link_lengths)
        goal_pos, _ = kd.forward_kinematics(self.goal, self.link_lengths)
        cartesian_dist = np.linalg.norm(config_pos[-1] - goal_pos[-1])

        # --- Joint space check ---
        joint_dist = self.config_distance(config, self.goal)

        # Debug print to see how close we are
        print(f"dist_cart: {cartesian_dist:.4f} m, dist_joint: {joint_dist:.4f} rad")

        # If either condition is satisfied, consider goal reached
        if cartesian_dist <= self.goal_tolerance or joint_dist <= self.goal_tolerance:
            return True
        return False

    def extract_path(self):
        """Extract path from start to goal"""
        try:
            # Find goal node (closest to goal within tolerance)
            goal_node_idx = None
            min_goal_dist = float('inf')

            for i, (config, _, _) in enumerate(self.tree):
                dist_to_goal = self.config_distance(config, self.goal)
                if dist_to_goal <= self.goal_tolerance and dist_to_goal < min_goal_dist:
                    min_goal_dist = dist_to_goal
                    goal_node_idx = i

            if goal_node_idx is None:
                return None

            # Backtrack from goal to start
            path = []
            current_idx = goal_node_idx

            while current_idx != -1:
                path.append(self.tree[current_idx][0].copy())
                current_idx = self.tree[current_idx][1]  # Move to parent

            path.reverse()
            return path
        except Exception as e:
            print(f"Error extracting path: {e}")
            return None

    def plan(self, goal_bias=0.1):
        """
        Main RRT* planning algorithm with real-time visualization.

        Args:
            goal_bias: Probability of sampling the goal instead of a random config.

        Returns:
            path: List of configurations from start to goal, or None if no path found.
        """
        print(f"Starting RRT* planning...")
        print(f"Start: {self.start}")
        print(f"Goal: {self.goal}")

        # Initialize real-time plot
        plt.ion()
        # PERFORMANCE: live matplotlib updates during the search cost real time
        # (draw_text / draw_path / sleep appeared in the profile). Enabled only
        # when the planner is constructed with live_view=True.
        fig = ax = None
        if getattr(self, 'live_view', False):
            fig, ax = vs.init_visualization(self.start, self.goal, self.link_lengths, self.obstacles)

        for iteration in range(self.max_iterations):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{self.max_iterations}")

            # -----------------------
            # 1. Sampling
            # -----------------------
            if random.random() < goal_bias:
                sample_config = self.goal.copy()
                # print(f"Sampled GOAL config: {sample_config}")
            else:
                sample_config = self.sample_random_config()
                # print(f"Sampled RANDOM config: {sample_config}")

            # -----------------------
            # 2. Nearest node
            # -----------------------
            nearest_idx, _ = self.find_nearest_node(sample_config)
            nearest_config = self.tree[nearest_idx][0]

            # -----------------------
            # 3. Steer towards sample
            # -----------------------
            new_config, _ = self.steer(nearest_config, sample_config)

            # -----------------------
            # 4. Check collision + validity
            # -----------------------
            # Single configuration checks
            collision_free = self.is_collision_free(new_config)
            limitations_ok = self.are_limitations_answered(new_config)

            # Path validation between configurations
            path_valid = self.is_path_valid(nearest_config, new_config) # New addition 03.09

            if not (collision_free and limitations_ok and path_valid):
                continue  # Reject this configuration

            # -----------------------
            # 5. Find near nodes for rewiring
            # -----------------------
            near_indices = self.find_near_nodes(new_config, self.rewire_radius)

            # -----------------------
            # 6. Choose best parent
            # -----------------------
            best_parent_idx = nearest_idx
            best_cost = self.get_path_cost(nearest_idx) + self.config_distance(nearest_config, new_config)

            for near_idx in near_indices:
                near_config = self.tree[near_idx][0]
                near_cost = self.get_path_cost(near_idx)
                try:
                    seg_time_cost = self.rrt_segment_cost(new_config, near_config)
                except ValueError:
                    # Physically infeasible edge -> reject this candidate parent,
                    # don't crash the whole planner. Expected in constrained RRT*.
                    continue
                potential_cost = near_cost + self.w_time * seg_time_cost + self.w_dist * self.config_distance(
                    near_config, new_config)
                if potential_cost < best_cost and self.is_path_valid(near_config, new_config):
                    best_parent_idx = near_idx
                    best_cost = potential_cost

            # -----------------------
            # 7. Add new node
            # -----------------------
            new_node_idx = len(self.tree)
            self.tree.append([new_config.copy(), best_parent_idx, best_cost])

            # -----------------------
            # 8. Visualization update
            # -----------------------
            if ax is not None:
                vs.update_visualization(ax, self.tree, self.link_lengths, self.obstacles)

            # -----------------------
            # 9. Rewire
            # -----------------------
            self.rewire_tree(new_node_idx, near_indices)

            # -----------------------
            # 10. Goal check
            # -----------------------
            if self.is_goal_reached(new_config):
                # CORRECTNESS FIX: the final edge from new_config to the goal was
                # previously appended WITHOUT validation. Because is_goal_reached
                # accepts on Cartesian OR joint distance, that last jump can be
                # large, so an unchecked edge could pass straight through an
                # obstacle or violate joint limits. The connection is now only
                # accepted if the segment is actually traversable.
                if not self.is_path_valid(new_config, self.goal):
                    continue

                print(f"\nGoal reached at iteration {iteration}!")
                print(f"Goal config found: {new_config}")

                goal_node_idx = len(self.tree)
                self.tree.append([self.goal.copy(), new_node_idx, best_cost])

                # Extract path
                self.path = self.extract_path()
                if self.path is not None:
                    print(f"Path found with {len(self.path)} waypoints")
                    print(f"Path cost: {self.get_path_cost(goal_node_idx):.3f}")
                    plt.ioff()
                    plt.show()
                    return self.path
                else:
                    print("Goal reached but failed to extract path!")
                    plt.ioff()
                    plt.show()
                    return None

        # -----------------------
        # 11. Failure
        # -----------------------
        print("Max iterations reached without finding path")
        plt.ioff()
        plt.show()
        return None

