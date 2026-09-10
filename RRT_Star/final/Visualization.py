## Libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import time

## Imported Functions
from Kinematics_and_Dynamics import forward_kinematics

## Real Time Visualization Functions

def draw_obstacles(ax, obstacles):
    """Draws obstacles on the given axis."""
    for obs in obstacles:
        if obs['type'] == 'circle':
            circle = patches.Circle(
                obs['center'], obs['radius'],
                facecolor='gray', edgecolor='black', alpha=0.5
            )
            ax.add_patch(circle)

        elif obs['type'] == 'rectangle':
            rect = patches.Rectangle(
                obs['bottom_left'], obs['width'], obs['height'],
                facecolor='gray', edgecolor='black', alpha=0.5
            )
            ax.add_patch(rect)


def init_visualization(start, goal, link_lengths, obstacles):
    plt.ion()  # Interactive mode ON
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.set_title('RRT* Tree Growing in Cartesian Space')

    # Draw obstacles
    draw_obstacles(ax, obstacles)

    # FK for start and goal
    start_pos, _ = forward_kinematics(start, link_lengths)
    goal_pos, _ = forward_kinematics(goal, link_lengths)

    # Plot start (green) and goal (red)
    ax.plot(start_pos[-1][0], start_pos[-1][1], 'go', label='Start')
    ax.plot(goal_pos[-1][0], goal_pos[-1][1], 'ro', label='Goal')

    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    plt.show()
    return fig, ax


def update_visualization(ax, tree, link_lengths, obstacles):
    # 1️⃣ Remove all old tree lines but keep Start & Goal
    for line in list(ax.lines):
        if line.get_label() not in ['Start', 'Goal']:
            line.remove()

    # 2️⃣ Remove all old obstacles
    for patch in list(ax.patches):
        patch.remove()

    # 3️⃣ Redraw obstacles
    draw_obstacles(ax, obstacles)

    # 4️⃣ Redraw tree edges in Cartesian space
    for config, parent_idx, _ in tree:
        if parent_idx != -1:
            parent = tree[parent_idx][0]
            parent_fk, _ = forward_kinematics(parent, link_lengths)
            child_fk, _ = forward_kinematics(config, link_lengths)
            parent_xy = parent_fk[-1]
            child_xy = child_fk[-1]
            ax.plot([parent_xy[0], child_xy[0]],
                    [parent_xy[1], child_xy[1]],
                    'y-', alpha=0.3)

    plt.pause(0.001)



## Route Visualization Functions

def plot_rrt_tree(planner, link_lengths, obstacles):
    """Plots the RRT* tree and solution path in Cartesian space."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw obstacles
    draw_obstacles(ax, obstacles)

    # Tree edges
    for config, parent_idx, _ in planner.tree:
        if parent_idx != -1:
            parent = planner.tree[parent_idx][0]
            parent_fk, _ = forward_kinematics(parent, link_lengths)
            child_fk, _ = forward_kinematics(config, link_lengths)
            parent_xy = parent_fk[-1]
            child_xy = child_fk[-1]
            ax.plot([parent_xy[0], child_xy[0]],
                    [parent_xy[1], child_xy[1]],
                    'y-', alpha=0.3)

    # Start & Goal
    start_fk, _ = forward_kinematics(planner.start, link_lengths)
    goal_fk, _ = forward_kinematics(planner.goal, link_lengths)
    ax.plot(start_fk[-1][0], start_fk[-1][1], 'go', label='Start')
    ax.plot(goal_fk[-1][0], goal_fk[-1][1], 'ro', label='Goal')

    # Solution Path
    if planner.path is not None:
        path_fk_points = []
        for config in planner.path:
            fk, _ = forward_kinematics(config, link_lengths)
            path_fk_points.append(fk[-1])
        path_fk_points = np.array(path_fk_points)
        ax.plot(path_fk_points[:, 0], path_fk_points[:, 1],
                'b-', linewidth=2, label='Solution Path')

    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.legend()
    ax.set_title('RRT* Tree and Solution Path (Cartesian Space)')
    ax.grid(True)
    ax.axis('equal')
    plt.show()

def animate_arm(path, link_lengths, obstacles):

    """

    :param path:
    :param link_lengths:
    :param obstacles:
    :return:
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    for obs in obstacles:
        if obs['type'] == 'circle':
            circle = plt.Circle(obs['center'], obs['radius'], color='r', alpha=0.5)
            ax.add_patch(circle)
        elif obs['type'] == 'rectangle':
            rect = plt.Rectangle(obs['bottom_left'], obs['width'], obs['height'], color='orange', alpha=0.5)
            ax.add_patch(rect)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('3DOF Arm Path Execution (Workspace)')

    arm_line, = ax.plot([], [], 'bo-', lw=4)

    def update(frame_idx):
        pos, _ = forward_kinematics(path[frame_idx], link_lengths)  # FK called here directly
        xs, ys = zip(*pos)
        arm_line.set_data(xs, ys)
        return arm_line,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=120, blit=True, repeat=False)
    plt.show()


