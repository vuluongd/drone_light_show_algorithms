from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
import os
import csv
CSV_OUTPUT = "sae_uet_surface.csv"


TEXT = "Vu Luong"
FONT_SIZE = 169
IMG_SIZE = (1600, 400)
MIN_DISTANCE = 5.1
NUM_DRONES = 600
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PLY_OUTPUT = "sae_uet_surface.ply"
# ==============================

# 1. Tạo ảnh chữ
img = Image.new('L', IMG_SIZE, color=0)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
bbox = draw.textbbox((0, 0), TEXT, font=font)
text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
text_pos = ((IMG_SIZE[0] - text_width) // 2, (IMG_SIZE[1] - text_height) // 2)
draw.text(text_pos, TEXT, fill=255, font=font)

# 2. Trích pixel trắng (màu chữ)
pixels = img.load()
white_pixels = [(x, y) for y in range(IMG_SIZE[1]) for x in range(IMG_SIZE[0]) if pixels[x, y] > 128]

# 3. Lọc viền ngoài: điểm trắng có ít nhất 1 hàng xóm đen
def is_edge(x, y):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < IMG_SIZE[0] and 0 <= ny < IMG_SIZE[1]:
                if pixels[nx, ny] < 128:
                    return True
    return False

edge_points = [(x, y) for x, y in white_pixels if is_edge(x, y)]
print(f"🧠 Tổng điểm viền chữ: {len(edge_points)}")

# 4. Scale về tọa độ thật (gốc ảnh ở giữa)
x_center, y_center = IMG_SIZE[0] / 2, IMG_SIZE[1] / 2
scaled_points = [((x - x_center), (y - y_center)) for x, y in edge_points]

# 5. Chọn 600 điểm cách nhau > 3m
selected = []
for px, py in scaled_points:
    too_close = any(math.hypot(px - x0, py - y0) < MIN_DISTANCE for x0, y0 in selected)
    if not too_close:
        selected.append((px, py))
    if len(selected) >= NUM_DRONES:
        break

if len(selected) < NUM_DRONES:
    raise RuntimeError(f"Không đủ {NUM_DRONES} điểm biên với khoảng cách tối thiểu {MIN_DISTANCE}m. Chỉ có {len(selected)} điểm.")

# 6. Hiển thị
xs, ys = zip(*selected)
plt.figure(figsize=(12, 5))
plt.scatter(xs, ys, s=10, c='red')
plt.gca().invert_yaxis()
plt.axis('equal')
plt.tight_layout()
plt.show()

with open(CSV_OUTPUT, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Drone_ID", "X (m)", "Y (m)", "Z (m)"])
    for idx, (x, y) in enumerate(selected, 1):
        writer.writerow([idx, round(x, 2), round(y, 2), 0.0])

print(f"✅  Đã lưu file CSV thành công: {CSV_OUTPUT}")
