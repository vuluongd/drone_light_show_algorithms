import math
import csv

MIN_DIST = 3.0
NUM_DRONES = 600
OUTPUT_CSV = "house_shape.csv"

# ========== Các hình cơ bản ==========
def line(p1, p2):
    """Trả về các điểm trên đoạn thẳng từ p1 đến p2"""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    steps = max(int(length // MIN_DIST), 1)
    return [(x1 + i * (x2 - x1) / steps, y1 + i * (y2 - y1) / steps) for i in range(steps + 1)]

def rectangle(x, y, w, h):
    return (
        line((x, y), (x + w, y)) +
        line((x + w, y), (x + w, y + h)) +
        line((x + w, y + h), (x, y + h)) +
        line((x, y + h), (x, y))
    )

def triangle(p1, p2, p3):
    return line(p1, p2) + line(p2, p3) + line(p3, p1)

# ========== Vẽ ngôi nhà ==========
points = []

# 1. Thân nhà (chữ nhật)
points += rectangle(-30, 0, 60, 40)

# 2. Mái nhà (tam giác)
points += triangle((-35, 40), (0, 70), (35, 40))

# 3. Cửa chính (chữ nhật rỗng)
points += rectangle(-10, 0, 20, 20)

# 4. Cửa sổ trái
points += rectangle(-25, 20, 10, 10)

# 5. Cửa sổ phải
points += rectangle(15, 20, 10, 10)

# ========== Loại trùng và cách tối thiểu ==========
def deduplicate(points, min_dist=MIN_DIST):
    result = []
    for x, y in points:
        if all(math.hypot(x - x0, y - y0) >= min_dist for x0, y0 in result):
            result.append((x, y))
        if len(result) >= NUM_DRONES:
            break
    return result

final_points = deduplicate(points)

# ========== Bổ sung điểm nếu thiếu ==========
if len(final_points) < NUM_DRONES:
    print(f"⚠️ Chỉ tạo được {len(final_points)} điểm từ hình. Bổ sung ngẫu nhiên vào mái.")
    from random import uniform
    extra_needed = NUM_DRONES - len(final_points)
    for _ in range(extra_needed):
        x = uniform(-30, 30)
        y = uniform(40, 70)
        final_points.append((x, y))

# ========== Xuất CSV ==========
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Drone_ID", "X (m)", "Y (m)", "Z (m)"])
    for i, (x, y) in enumerate(final_points, 1):
        writer.writerow([i, round(x, 2), round(y, 2), 0.0])

print(f"✅ Đã tạo đội hình 'Ngôi Nhà' với {len(final_points)} drone → {OUTPUT_CSV}")
