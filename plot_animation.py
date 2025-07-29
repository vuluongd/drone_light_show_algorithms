import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

file_path = "drone_paths_output_1.xlsx"
xls = pd.ExcelFile(file_path)
chosen_sheets = input("Nhập sheet ").split( )
chosen_sheets = [sheet.strip() for sheet in chosen_sheets if sheet.strip() in xls.sheet_names]

trajectories = {}
for sheet in chosen_sheets:
    df = xls.parse(sheet)
    trajectories[sheet] = {
        'x': df['X'].values,
        'y': df['Y'].values,
        'z': df['Z'].values
    }

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Animation các sheet')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid(True)

all_x = [pt for traj in trajectories.values() for pt in traj['x']]
all_y = [pt for traj in trajectories.values() for pt in traj['y']]
all_z = [pt for traj in trajectories.values() for pt in traj['z']]
ax.set_xlim(min(all_x)-1, max(all_x)+1)
ax.set_ylim(min(all_y)-1, max(all_y)+1)
ax.set_zlim(min(all_z)-1, max(all_z)+1)

lines = {}
points = {}
colors = ['r', 'g', 'b', 'm', 'c', 'orange', 'purple']

for i, sheet in enumerate(chosen_sheets):
    color = colors[i % len(colors)]
    (line,) = ax.plot([], [], [], '-', color=color, label=sheet)
    (point,) = ax.plot([], [], [], 'o', color=color)
    lines[sheet] = line
    points[sheet] = point

def init():
    for sheet in chosen_sheets:
        lines[sheet].set_data([], [])
        lines[sheet].set_3d_properties([])
        points[sheet].set_data([], [])
        points[sheet].set_3d_properties([])
    return list(lines.values()) + list(points.values())

def update(frame):
    for sheet in chosen_sheets:
        x = trajectories[sheet]['x']
        y = trajectories[sheet]['y']
        z = trajectories[sheet]['z']
        if frame < len(x):
            lines[sheet].set_data(x[:frame], y[:frame])
            lines[sheet].set_3d_properties(z[:frame])
            points[sheet].set_data([x[frame-1]], [y[frame-1]])
            points[sheet].set_3d_properties([z[frame-1]])
    return list(lines.values()) + list(points.values())

max_frames = max(len(t['x']) for t in trajectories.values())
ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, interval=200, blit=False)

ax.legend()
ani.save("animation_output.mp4", writer='ffmpeg', fps=25)
plt.show()
