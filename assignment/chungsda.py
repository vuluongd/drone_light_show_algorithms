import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

file_path = "~/drone_light_show_algorithm/formation4.xlsx"
xls = pd.ExcelFile(os.path.expanduser(file_path))
scene_names = ["scene_1", "scene_2", "scene_3", "scene_4"]
scenes = [list(zip(xls.parse(name)["x"], xls.parse(name)["y"], xls.parse(name)["z"])) for name in scene_names]

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def hungarian_distances(start_points, end_points):
    n = len(start_points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = euclidean_distance(start_points[i], end_points[j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    distances = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    return distances

all_distances = []
total_distances = []

for i in range(3):
    start = scenes[i]
    end = scenes[i+1]
    distances = hungarian_distances(start, end)
    all_distances.append(distances)
    total_distances.append(sum(distances))

num_drones = len(all_distances[0])
drone_total_distances = []
for drone_idx in range(num_drones):
    total_for_drone = sum(all_distances[scene][drone_idx] for scene in range(3))
    drone_total_distances.append(total_for_drone)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

scene_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
for i in range(3):
    distances = all_distances[i]
    mean_distance = np.mean(distances)
    drone_ids = list(range(1, len(distances) + 1))

    ax = scene_axes[i]
    ax.scatter(drone_ids, distances, color='blue', label='Distances')
    ax.hlines(mean_distance, xmin=1, xmax=len(drone_ids), color='red', linestyle='--', 
              label=f"Mean: {mean_distance:.2f}")
    ax.set_title(f"Scene {i+1} → Scene {i+2}", fontsize=12)
    ax.set_xlabel("Drone Index")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True)

ax_total = axes[1, 1]
drone_ids = list(range(1, num_drones + 1))
mean_total_distance = np.mean(drone_total_distances)

ax_total.scatter(drone_ids, drone_total_distances, color='green', label='Total Distances')
ax_total.hlines(mean_total_distance, xmin=1, xmax=num_drones, color='red', linestyle='--', 
                label=f"Mean: {mean_total_distance:.2f}")
ax_total.set_title("Tổng Distance của từng Drone (All Scenes)", fontsize=12)
ax_total.set_xlabel("Drone Index")
ax_total.set_ylabel("Total Distance")
ax_total.legend()
ax_total.grid(True)

plt.suptitle("Hungarian Algorithm - Drone Distance Assignments qua các cảnh", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("\Thống kê tổng kết:")
for i in range(3):
    print(f"Scene {i+1} → Scene {i+2}: Tổng distance = {total_distances[i]:.2f}")
print(f"Tổng distance tất cả scenes: {sum(total_distances):.2f}")
print(f"Trung bình distance/drone: {mean_total_distance:.2f}")
