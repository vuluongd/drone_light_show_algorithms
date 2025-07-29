import numpy as np
import pandas as pd

def generate_grid_formation(n_drones, spacing=2.5):
    """
    Tạo đội hình lưới cho n drone với khoảng cách an toàn
    """
    # Tính số hàng và cột cần thiết
    grid_size = int(np.ceil(np.sqrt(n_drones)))
    
    start_points = []
    end_points = []
    
    # Tăng không gian cho điểm đến
    end_spacing = spacing * 3  # Tăng khoảng cách giữa các điểm đến
    end_center = grid_size * end_spacing / 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(start_points) < n_drones:
                # Vị trí xuất phát trên mặt phẳng z=0
                start_points.append((
                    i * spacing,
                    j * spacing,
                    0
                ))
                
                # Vị trí kết thúc: tạo một lưới rộng hơn ở độ cao khác nhau
                end_points.append((
                    i * end_spacing,
                    j * end_spacing,
                    30 + (i + j) * spacing  # Tạo độ cao khác nhau để tránh va chạm
                ))
    
    return start_points, end_points

def generate_cube_formation(n_drones, spacing=2.5):
    """
    Tạo đội hình khối lập phương
    """
    cube_size = int(np.ceil(np.cbrt(n_drones)))
    end_spacing = spacing * 3
    
    start_points = []
    end_points = []
    
    for x in range(cube_size):
        for y in range(cube_size):
            for z in range(cube_size):
                if len(start_points) < n_drones:
                    # Điểm xuất phát trong mặt phẳng
                    start_points.append((
                        x * spacing,
                        y * spacing,
                        0
                    ))
                    
                    # Điểm đến trong không gian 3D
                    end_points.append((
                        x * end_spacing + 50,
                        y * end_spacing + 50,
                        z * end_spacing + 50
                    ))
    
    return start_points, end_points

def generate_cylinder_formation(n_drones, spacing=2.5):
    """
    Tạo đội hình hình trụ
    """
    # Tính số lớp và số drone trên mỗi vòng tròn
    n_layers = int(np.sqrt(n_drones / 2))
    n_per_circle = int(np.ceil(n_drones / n_layers))
    
    start_points = []
    end_points = []
    end_spacing = spacing * 3
    radius = n_per_circle * end_spacing / (2 * np.pi)
    
    for layer in range(n_layers):
        for i in range(n_per_circle):
            if len(start_points) < n_drones:
                # Điểm xuất phát trong lưới
                start_points.append((
                    (i % int(np.sqrt(n_drones))) * spacing,
                    (i // int(np.sqrt(n_drones))) * spacing,
                    0
                ))
                
                # Điểm đến trong hình trụ
                angle = 2 * np.pi * i / n_per_circle
                end_points.append((
                    50 + radius * np.cos(angle),
                    50 + radius * np.sin(angle),
                    layer * end_spacing + 50
                ))
    
    return start_points, end_points

def validate_formation(start_points, end_points, min_spacing=2.0):
    """
    Kiểm tra tính hợp lệ của đội hình
    """
    if len(start_points) != len(end_points):
        raise ValueError("Số lượng điểm bắt đầu và kết thúc không bằng nhau")
    
    # Kiểm tra khoảng cách tối thiểu
    start_min_dist = min_distance(start_points)
    end_min_dist = min_distance(end_points)
    
    print(f"Khoảng cách tối thiểu tại điểm xuất phát: {start_min_dist}")
    print(f"Khoảng cách tối thiểu tại điểm đến: {end_min_dist}")
    
    if start_min_dist < min_spacing:
        raise ValueError(f"Khoảng cách giữa các điểm xuất phát quá nhỏ: {start_min_dist}")
    if end_min_dist < min_spacing:
        raise ValueError(f"Khoảng cách giữa các điểm kết thúc quá nhỏ: {end_min_dist}")
    
    return True

def min_distance(points):
    """Tính khoảng cách nhỏ nhất giữa các điểm"""
    min_dist = float('inf')
    points_array = np.array(points)
    
    for i in range(len(points)):
        # Tính toán khoảng cách từ điểm i đến tất cả các điểm còn lại
        distances = np.sqrt(np.sum((points_array[i+1:] - points_array[i])**2, axis=1))
        if len(distances) > 0:
            min_dist = min(min_dist, np.min(distances))
    
    return min_dist

def save_formation_to_excel(start_points, end_points, formation_name, output_file):
    """
    Lưu đội hình vào file Excel
    """
    # Tạo DataFrames
    start_df = pd.DataFrame(start_points, columns=['x', 'y', 'z'])
    end_df = pd.DataFrame(end_points, columns=['x', 'y', 'z'])
    
    # Thêm cột drone_id
    start_df.insert(0, 'drone_id', range(len(start_points)))
    end_df.insert(0, 'drone_id', range(len(end_points)))
    
    # Lưu vào Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        start_df.to_excel(writer, sheet_name='Start_Points', index=False)
        end_df.to_excel(writer, sheet_name='End_Points', index=False)
        
        # Tạo sheet thống kê
        stats = {
            'Metric': ['Number of drones', 
                      'Formation type',
                      'Min distance at start',
                      'Min distance at end',
                      'Max X coordinate',
                      'Max Y coordinate',
                      'Max Z coordinate'],
            'Value': [
                len(start_points),
                formation_name,
                min_distance(start_points),
                min_distance(end_points),
                max(max(p[0] for p in start_points), max(p[0] for p in end_points)),
                max(max(p[1] for p in start_points), max(p[1] for p in end_points)),
                max(max(p[2] for p in start_points), max(p[2] for p in end_points))
            ]
        }
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

if __name__ == "__main__":
    # Tham số
    n_drones = 50
    min_spacing = 2.5
    
    # Thử các đội hình khác nhau
    formations = {
        'grid': generate_grid_formation,
        'cube': generate_cube_formation,
        'cylinder': generate_cylinder_formation
    }
    
    for formation_name, formation_func in formations.items():
        try:
            print(f"\nTạo đội hình {formation_name}...")
            start_points, end_points = formation_func(n_drones, min_spacing)
            
            print(f"Kiểm tra tính hợp lệ của đội hình {formation_name}...")
            validate_formation(start_points, end_points, min_spacing)
            
            output_file = f'drone_formation_500_{formation_name}.xlsx'
            save_formation_to_excel(start_points, end_points, formation_name, output_file)
            print(f"Đã lưu đội hình {formation_name} vào {output_file}")
            
        except Exception as e:
            print(f"Lỗi khi tạo đội hình {formation_name}: {str(e)}")
            continue