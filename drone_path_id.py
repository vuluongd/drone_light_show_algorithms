import numpy as np
import pandas as pd

n_drones = 600
step = 4 

indices = np.arange(0, n_drones)
phi = np.arccos(1 - 2 * (indices + 0.5) / n_drones)
theta = np.pi * (1 + 5**0.5) * indices

R = 40  
x_sphere = R * np.sin(phi) * np.cos(theta)
y_sphere = R * np.sin(phi) * np.sin(theta)
z_sphere = R * np.cos(phi)

df_start = pd.DataFrame({
    'id': indices,
    'X': x_sphere,
    'Y': y_sphere,
    'Z': z_sphere
})


nx, ny, nz = 10, 10, 6 
x = np.arange(0, nx * step, step)
y = np.arange(0, ny * step, step)
z = np.arange(0, nz * step, step)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


df_end = pd.DataFrame(points, columns=['X', 'Y', 'Z']).head(n_drones)
df_end.insert(0, 'id', indices)


with pd.ExcelWriter("drone_formations_sphere_box.xlsx") as writer:
    df_start.to_excel(writer, sheet_name="Start_Sphere", index=False)
    df_end.to_excel(writer, sheet_name="End_Box", index=False)

print("✅ Đã tạo và lưu đội hình hình cầu & hình hộp vào 'drone_formations_sphere_box.xlsx'")
