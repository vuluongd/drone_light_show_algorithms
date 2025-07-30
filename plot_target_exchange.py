import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linear_sum_assignment

file_path_2 = "~/drone_light_show_algorithm/formation3.xlsx"
xls = pd.ExcelFile(os.path.expanduser(file_path_2))
start_points_df = xls.parse("Start_Points")
end_points_df = xls.parse("End_Points")
start_points = list(zip(start_points_df["x"], start_points_df["y"], start_points_df["z"]))
end_points = list(zip(end_points_df["x"], end_points_df["y"], end_points_df["z"]))

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def hungarian_algorithm(start_points, end_points):
    n = len(start_points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = euclidean_distance(start_points[i], end_points[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i, j in zip(row_ind, col_ind):
        if euclidean_distance(start_points[i], end_points[j]) < 1e-6:
            raise ValueError("Invalid assignment: Start and end points coincide")
    return list(zip(row_ind, col_ind))

def full_algorithm(start_points, end_points):
    n = len(start_points)
    min_safe_distance = 2.5
    start_violations = [(i, j) for i in range(n) for j in range(i + 1, n)
                        if euclidean_distance(start_points[i], start_points[j]) < min_safe_distance]
    end_violations = [(i, j) for i in range(n) for j in range(i + 1, n)
                      if euclidean_distance(end_points[i], end_points[j]) < min_safe_distance]
    if start_violations or end_violations:
        raise ValueError("Vi phạm khoảng cách an toàn tối thiểu 2.5m")

    assignments = hungarian_algorithm(start_points, end_points)
    paths = []
    for drone_idx, target_idx in assignments:
        start = np.array(start_points[drone_idx])
        end = np.array(end_points[target_idx])
        paths.append((drone_idx, [start, end]))
    return paths

paths = full_algorithm(start_points, end_points)

selected = input("\nphan cong hungarian cua cac drone : ")
selected_indices = [int(idx) for idx in selected.split() if idx.isdigit()]

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
colors = plt.cm.jet(np.linspace(0, 1, len(paths)))

for i, (drone_idx, path) in enumerate(paths):
    if drone_idx in selected_indices:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[i], label=f'Hungarian {drone_idx}')
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='green', marker='o', s=100)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='red', marker='x', s=100)

file_path = "drone_paths_output_1.xlsx"
xls = pd.ExcelFile(file_path)

chosen_sheets = input("\nNhập sheet: ").split()
chosen_sheets = [sheet.strip() for sheet in chosen_sheets if sheet.strip() in xls.sheet_names]

trajectories = {}
for sheet in chosen_sheets:
    df = xls.parse(sheet)
    trajectories[sheet] = {
        'x': df['X'].values,
        'y': df['Y'].values,
        'z': df['Z'].values
    }

colors_extra = ['r', 'g', 'b', 'm', 'c', 'orange', 'purple']
for i, sheet in enumerate(chosen_sheets):
    color = colors_extra[i % len(colors_extra)]
    x = trajectories[sheet]['x']
    y = trajectories[sheet]['y']
    z = trajectories[sheet]['z']
    ax.plot(x, y, z, label=f"Trajectory {sheet}", color=color, linestyle='--')
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, marker='o', s=80)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
