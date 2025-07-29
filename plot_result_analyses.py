import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_path_file = "drone_paths_output.xlsx"
xls = pd.ExcelFile(input_path_file)

summary_df = pd.read_excel(input_path_file, sheet_name="Summary")
total_iterations = summary_df["Total Iterations"][0]
n_drones = summary_df["Number of Drones"][0]
drone_points = [summary_df[f"Drone_{i} Points"][0] for i in range(n_drones)]

print(f"Total iterations from Summary: {total_iterations}")
print(f"Drone points: {drone_points}")

drone_data = {}
for i in range(n_drones):
    sheet_name = f"Drone_{i}"
    drone_df = pd.read_excel(input_path_file, sheet_name=sheet_name)
    drone_data[i] = drone_df[["X", "Y", "Z"]].values

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

min_distances = []
for step in range(total_iterations):
    positions = []
    for i in range(n_drones):
        if step < len(drone_data[i]):
            positions.append(drone_data[i][step])  
        else:
            positions.append(drone_data[i][-1])  
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = euclidean_distance(positions[i], positions[j])
            if dist < min_dist:
                min_dist = dist
    min_distances.append(min_dist)

dt = 1 / 25  # Thời gian giữa các bước (25 Hz)
velocities = {i: [] for i in range(n_drones)}
accelerations = {i: [] for i in range(n_drones)}

for i in range(n_drones):
    path = drone_data[i]
    
    for step in range(1, len(path)):
        v = euclidean_distance(path[step], path[step-1]) / dt
        velocities[i].append(v)
    
    for step in range(1, len(velocities[i])):
        a = abs(velocities[i][step] - velocities[i][step-1]) / dt
        accelerations[i].append(a)
        
    while len(velocities[i]) < total_iterations - 1:
        velocities[i].append(velocities[i][-1] if velocities[i] else 0.0)
    while len(accelerations[i]) < total_iterations - 2:
        accelerations[i].append(accelerations[i][-1] if accelerations[i] else 0.0)

plt.style.use('default')

plt.figure(figsize=(10, 6))
for i in range(n_drones):
    plt.plot(range(1, len(velocities[i]) + 1), velocities[i], alpha=0.5, label=f"Drone {i}" if i < 5 else None)
plt.axhline(y=3, color='r', linestyle='--', label='Max Speed (3 m/s)')
plt.xlabel("Waypoints")
plt.ylabel("Speed (m/s)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("speed_magnitude.png")
plt.close()

plt.figure(figsize=(10, 6))
for i in range(n_drones):
    acc_plot = [min(a, 3.0) for a in accelerations[i]]
    plt.plot(range(2, len(acc_plot) + 2), acc_plot, alpha=0.5, label=f"Drone {i}" if i < 5 else None)
plt.axhline(y=3, color='r', linestyle='--', label='Max Acceleration (3 m/s²)')
plt.xlabel("Waypoints")
plt.ylabel("Acceleration (m/s²)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("acceleration_magnitude.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(range(total_iterations), min_distances, color='b')
plt.axhline(y=2.5, color='r', linestyle='--', label='Min Safe Distance (2.5 m)')
plt.xlabel("Waypoints")
plt.ylabel("Distance (m)")
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("min_distance.png")
plt.close()

print("speed.png, acceleration.png, min_distance.png")