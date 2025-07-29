import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

file_path = "~/drone_light_show_algorithm/formation4.xlsx"
xls = pd.ExcelFile(file_path)
scene_1 = xls.parse("scene_1")
scene_2 = xls.parse("scene_2")
scene_3 = xls.parse("scene_3")
scene_4 = xls.parse("scene_4")
fig = plt.figure(figsize=(12, 10))  
ax = fig.add_subplot(111, projection='3d')

ax.scatter(scene_1['x'], scene_1['y'], scene_1['z'], c='blue', label='scene_1', alpha=0.6)
ax.scatter(scene_2['x'], scene_2['y'], scene_2['z'], c='red', label='scene_2', alpha=0.6)
ax.scatter(scene_3['x'], scene_3['y'], scene_3['z'], c='green', label= 'scene_3', alpha=0.6)
ax.scatter(scene_4['x'], scene_4['y'], scene_4['z'], c='orange', label='scene_4', alpha=0.6)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.view_init(-150,-90,0)
ax.set_title("Drone Formations")
ax.legend()
ax.set_zlim(0,70)

plt.tight_layout()
plt.show()
