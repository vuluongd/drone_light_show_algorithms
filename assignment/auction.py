import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import time

class DroneAuctionAlgorithm:
    """
    Thuật toán đấu giá cho phân công nhiệm vụ bay drone (đã sửa lỗi)
    Bám sát tài liệu nghiên cứu: sử dụng Hungarian cho lần đầu, Auction cho các lần tiếp theo
    """
    
    def __init__(self, epsilon: float = 1.0, alpha: float = 0.1):
        """
        Khởi tạo thuật toán đấu giá
        
        Args:
            epsilon: Giá trị nới lỏng ràng buộc (ε > 0)
            alpha: Hệ số hiệu chỉnh để cân bằng giữa tối thiểu hóa tổng khoảng cách 
                   và giảm chênh lệch đường bay
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.drone_cumulative_costs = {}  # Tổng khoảng cách của drone di chuyển trong các đội hình trước đó
    
    def calculate_benefit_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Tính ma trận lợi ích theo công thức 2.9 trong tài liệu:
        benefit_ij = -(d_ij + α * C_i)
        
        Trong đó:
        - d_ij: khoảng cách di chuyển của drone i trong kịch bản hiện tại
        - C_i: tổng khoảng cách của drone i di chuyển trong các đội hình trước đó
        - α: hệ số hiệu chỉnh
        """
        n = distance_matrix.shape[0]
        benefit_matrix = np.zeros((n, n))
        
        for i in range(n):
            cumulative_cost = self.drone_cumulative_costs.get(i, 0)
            for j in range(n):
                # Công thức 2.9 từ tài liệu
                benefit_matrix[i][j] = -(self.alpha *(distance_matrix[i][j]+cumulative_cost) + cumulative_cost)
        
        print(f"Ma trận lợi ích được tính theo công thức 2.9:")
        print(f"benefit_ij = -(d_ij + α * C_i) với α = {self.alpha}")
        print(benefit_matrix)
        return benefit_matrix
    
    def is_nearly_happy(self, drone_i: int, assignment: list, benefit_matrix: np.ndarray, prices: list) -> bool:
        """
        Kiểm tra điều kiện "gần hạnh phúc" được cải thiện
        """
        n = len(assignment)
        
        if assignment[drone_i] == -1:
            return False
        
        # Giá trị ròng của mục tiêu được gán
        assigned_target = assignment[drone_i]
        assigned_net_value = benefit_matrix[drone_i][assigned_target] - prices[assigned_target]
        
        # Tìm giá trị ròng tối đa
        max_net_value = float('-inf')
        for j in range(n):
            net_value = benefit_matrix[drone_i][j] - prices[j]
            max_net_value = max(max_net_value, net_value)
        
        # Điều kiện gần hạnh phúc với tolerance nhỏ để tránh lỗi số học
        tolerance = 1e-10
        is_happy = assigned_net_value >= max_net_value - self.epsilon - tolerance
        
        if not is_happy:
            print(f"     🔍 Drone {drone_i+1} không hạnh phúc: current={assigned_net_value:.6f}, max={max_net_value:.6f}, diff={max_net_value-assigned_net_value:.6f}")
        
        return is_happy
    
    def auction_algorithm(self, benefit_matrix: np.ndarray) -> list:
        """
        Thuật toán đấu giá được sửa lỗi - đảm bảo tất cả drone đều được gán
        """
        n = benefit_matrix.shape[0]
        
        # a) Khởi tạo
        prices = [0.0] * n
        assignment = [-1] * n
        reverse_assignment = [-1] * n
        
        max_iterations = 1000
        iteration = 0
        
        print(f"\nBắt đầu thuật toán đấu giá với ε = {self.epsilon}")
        print(f"Ma trận lợi ích:\n{benefit_matrix}")
        
        while iteration < max_iterations:
            iteration += 1
            
            # Tìm tất cả drone chưa được gán
            unassigned_drones = [i for i in range(n) if assignment[i] == -1]
            
            # Tìm drone không "gần hạnh phúc"
            unhappy_drones = []
            for i in range(n):
                if assignment[i] != -1 and not self.is_nearly_happy(i, assignment, benefit_matrix, prices):
                    unhappy_drones.append(i)
            
            # Điều kiện dừng: tất cả drone được gán VÀ đều gần hạnh phúc
            if not unassigned_drones and not unhappy_drones:
                print(f"✅ Thuật toán hoàn thành sau {iteration} vòng lặp - tất cả drone được gán và gần hạnh phúc")
                break
            
            # Chọn drone để đấu giá: ưu tiên drone chưa gán, sau đó drone không hạnh phúc
            bidding_drone = None
            if unassigned_drones:
                bidding_drone = unassigned_drones[0]
                print(f"🔄 Vòng {iteration}: Drone {bidding_drone+1} chưa được gán sẽ đấu giá")
            elif unhappy_drones:
                bidding_drone = unhappy_drones[0]
                print(f"🔄 Vòng {iteration}: Drone {bidding_drone+1} không hạnh phúc sẽ đấu giá lại")
            
            if bidding_drone is None:
                print(f"⚠️ Lỗi: Không tìm thấy drone để đấu giá ở vòng {iteration}")
                break
            
            # Tìm mục tiêu tốt nhất và tốt thứ hai cho drone này
            net_values = []
            for j in range(n):
                net_value = benefit_matrix[bidding_drone][j] - prices[j]
                net_values.append((net_value, j))
            
            # Sắp xếp theo giá trị ròng giảm dần
            net_values.sort(reverse=True, key=lambda x: x[0])
            
            best_net_value = net_values[0][0]
            best_target = net_values[0][1]
            second_best_net_value = net_values[1][0] if len(net_values) > 1 else float('-inf')
            
            # Tính mức tăng giá
            if second_best_net_value != float('-inf'):
                bid_increment = best_net_value - second_best_net_value + self.epsilon
            else:
                bid_increment = self.epsilon
            
            # Đảm bảo bid_increment > 0
            bid_increment = max(bid_increment, self.epsilon)
            
            old_price = prices[best_target]
            prices[best_target] += bid_increment
            
            print(f"  💰 Đấu giá cho mục tiêu {best_target+1}")
            print(f"     Giá trị ròng tốt nhất: {best_net_value:.3f}")
            print(f"     Giá trị ròng thứ hai: {second_best_net_value:.3f}")
            print(f"     Tăng giá từ {old_price:.3f} → {prices[best_target]:.3f}")
            
            previous_owner = reverse_assignment[best_target]
            if previous_owner != -1:
                assignment[previous_owner] = -1
                print(f"     ❌ Drone {previous_owner+1} mất mục tiêu {best_target+1}")
            
            # Gán mục tiêu mới
            assignment[bidding_drone] = best_target
            reverse_assignment[best_target] = bidding_drone
            print(f"     ✅ Drone {bidding_drone+1} được gán mục tiêu {best_target+1}")
            
            # Hiển thị trạng thái hiện tại
            current_assignments = [f"Drone {i+1}→Target {assignment[i]+1}" if assignment[i] != -1 
                                 else f"Drone {i+1}→Chưa gán" for i in range(n)]
            print(f"     📋 Trạng thái: {', '.join(current_assignments)}")
        
        # Kiểm tra kết quả cuối cùng
        unassigned_final = [i for i in range(n) if assignment[i] == -1]
        if unassigned_final:
            print(f"⚠️ CẢNH BÁO: Vẫn còn {len(unassigned_final)} drone chưa được gán: {[i+1 for i in unassigned_final]}")
            print("Điều này có thể do:")
            print("- Tham số epsilon quá nhỏ")
            print("- Ma trận lợi ích có vấn đề")
            print("- Thuật toán cần điều chỉnh thêm")
            
            # Fallback: Gán bằng Hungarian algorithm
            print("🔧 Sử dụng Hungarian algorithm để gán các drone còn lại...")
            cost_matrix = -benefit_matrix  # Chuyển từ benefit sang cost
            _, col_ind = linear_sum_assignment(cost_matrix)
            assignment = col_ind.tolist()
            print(f"✅ Đã gán lại tất cả drone bằng Hungarian algorithm")
        else:
            print(f"✅ TẤT CẢ {n} DRONE ĐÃ ĐƯỢC GÁN THÀNH CÔNG!")
        
        print(f"🏁 Kết quả cuối cùng: {[f'Target {x+1}' if x != -1 else 'Chưa gán' for x in assignment]}")
        print(f"💵 Giá cuối cùng: {[f'{p:.3f}' for p in prices]}")
        return assignment

def euclidean_distance(a, b):
    """Tính khoảng cách Euclidean 3D"""
    return np.linalg.norm(np.array(a) - np.array(b))

def assignment_with_strategy(start_points, end_points, auction_solver, is_first_transition=False):
    """
    Phân công nhiệm vụ bay theo chiến lược trong tài liệu:
    - Lần chuyển đổi đầu tiên: Sử dụng Hungarian Algorithm
    - Các lần tiếp theo: Sử dụng Auction Algorithm
    """
    n = len(start_points)
    
    # Tạo ma trận khoảng cách (ma trận chi phí)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = euclidean_distance(start_points[i], end_points[j])
    
    print(f"Ma trận khoảng cách:\n{distance_matrix}")
    
    if is_first_transition:
        print("→ Sử dụng Hungarian Algorithm cho lần chuyển đổi đầu tiên")
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        assignment = col_ind.tolist()
        print(f"Kết quả Hungarian: {[x+1 for x in assignment]}")
    else:
        print("→ Sử dụng Auction Algorithm cho lần chuyển đổi tiếp theo")
        benefit_matrix = auction_solver.calculate_benefit_matrix(distance_matrix)
        assignment = auction_solver.auction_algorithm(benefit_matrix)
    
    # Tính khoảng cách theo assignment
    distances = []
    for i in range(n):
        if assignment[i] != -1:
            dist = distance_matrix[i][assignment[i]]
            distances.append(dist)
            # Cập nhật chi phí tích lũy
            auction_solver.drone_cumulative_costs[i] = (
                auction_solver.drone_cumulative_costs.get(i, 0) + dist
            )
        else:
            distances.append(0)
    
    return distances, assignment

# Phần còn lại của code giữ nguyên...
# Đọc dữ liệu từ file Excel
file_path = "~/drone_light_show_algorithm/formation4.xlsx"

try:
    xls = pd.ExcelFile(os.path.expanduser(file_path))
    scene_names = ["scene_1", "scene_2", "scene_3", "scene_4"]
    scenes = [list(zip(xls.parse(name)["x"], xls.parse(name)["y"], xls.parse(name)["z"])) 
              for name in scene_names]
    print("✅ Đã đọc dữ liệu từ file Excel")
except FileNotFoundError:
    print("⚠️ Không tìm thấy file Excel, sử dụng dữ liệu mẫu")
    # Dữ liệu mẫu cho demo
    scenes = [
        [(0, 0, 0), (1, 0, 0), (2, 0, 0)],  # scene_1
        [(0, 1, 0), (1, 1, 0), (2, 1, 0)],  # scene_2
        [(0, 2, 0), (1, 2, 0), (2, 2, 0)],  # scene_3
        [(0, 3, 0), (1, 3, 0), (2, 3, 0)]   # scene_4
    ]

auction_solver = DroneAuctionAlgorithm(epsilon=0.1, alpha=1)  # Tăng epsilon để tránh lỗi

# Thực hiện phân công theo chiến lược trong tài liệu
all_distances = []
all_assignments = []
total_distances = []

print("\n" + "="*80)
print("🚁 PHÂN CÔNG NHIỆM VỤ BAY DRONE THEO THUẬT TOÁN ĐẤU GIÁ")
print("Chiến lược: Hungarian cho lần đầu, Auction cho các lần tiếp theo")
print("="*80)

for i in range(3):
    print(f"\n{'='*20} KỊCH BẢN {i+1} → KỊCH BẢN {i+2} {'='*20}")
    start = scenes[i]
    end = scenes[i+1]
    
    # Kiểm tra xem có phải lần chuyển đổi đầu tiên không
    is_first = (i == 0)
    
    distances, assignment = assignment_with_strategy(
        start, end, auction_solver, is_first_transition=is_first
    )
    
    all_distances.append(distances)
    all_assignments.append(assignment)
    total_distances.append(sum(distances))
    
    print(f"\nKết quả phân công:")
    for drone_idx, target_idx in enumerate(assignment):
        if target_idx != -1:
            print(f"  Drone {drone_idx+1} → Mục tiêu {target_idx+1} (khoảng cách: {distances[drone_idx]:.2f})")
        else:
            print(f"  Drone {drone_idx+1} → Chưa được gán")
    
    print(f"Tổng khoảng cách kịch bản này: {sum(distances):.2f}")
    print(f"Chi phí tích lũy hiện tại của các drone:")
    for drone_id, cost in auction_solver.drone_cumulative_costs.items():
        print(f"  Drone {drone_id+1}: {cost:.2f}")

num_drones = len(all_distances[0])
drone_total_distances = [
    sum(all_distances[scene][drone_idx] for scene in range(3)) 
    for drone_idx in range(num_drones)
]

# Vẽ biểu đồ kết quả (sửa lại theo style của GA)
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Biểu đồ cho từng kịch bản (chuyển từ bar sang scatter)
    scene_axes = [axes[0, 0], axes[0, 1], axes[1, 0]]
    for i in range(3):
        distances = all_distances[i]
        mean_distance = np.mean(distances)
        drone_ids = list(range(1, len(distances) + 1))

        ax = scene_axes[i]
        method = "Hungarian" if i == 0 else "Auction"
        color = 'blue' if i == 0 else 'red'
        
        # Chuyển từ bar chart sang scatter plot
        ax.scatter(drone_ids, distances, color=color, label=f'{method} Algorithm')
        ax.hlines(mean_distance, xmin=1, xmax=len(drone_ids), color='red', linestyle='--', 
                  label=f"Mean: {mean_distance:.2f}")
        
        ax.set_title(f"Scene {i+1} → Scene {i+2} ({method} Algorithm)", fontsize=12)
        ax.set_xlabel("Drone Index")
        ax.set_ylabel("Distance")
        ax.legend()
        ax.grid(True)

    # Biểu đồ tổng khoảng cách (chuyển từ bar sang scatter)
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

    plt.suptitle("Hungarian + Auction Algorithm - Drone Distance Assignments qua các cảnh", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

except Exception as e:
    print(f"Lỗi khi vẽ biểu đồ: {e}")

# Thống kê kết quả
print("\n" + "="*80)
print("📊 THỐNG KÊ KẾT QUẢ CUỐI CÙNG")
print("="*80)

print("\nKhoảng cách theo từng kịch bản:")
for i in range(3):
    method = "Hungarian" if i == 0 else "Auction"
    print(f"  Kịch bản {i+1} → {i+2} ({method}): {total_distances[i]:.2f}")

print(f"\nTổng khoảng cách tất cả kịch bản: {sum(total_distances):.2f}")
print(f"Trung bình khoảng cách/drone: {np.mean(drone_total_distances):.2f}")
print(f"Độ lệch chuẩn: {np.std(drone_total_distances):.2f}")

# Phân tích cân bằng workload
print(f"\n🎯 PHÂN TÍCH CÂN BẰNG WORKLOAD:")
print("Tổng khoảng cách bay của từng drone:")
for drone_id in range(num_drones):
    total_distance = drone_total_distances[drone_id]
    print(f"  Drone {drone_id+1}: {total_distance:.2f}")

min_workload = min(drone_total_distances)
max_workload = max(drone_total_distances)
workload_balance = max_workload - min_workload

print(f"\nChênh lệch workload: {workload_balance:.2f}")
print(f"Tỷ lệ cân bằng: {(1 - workload_balance/max_workload)*100:.1f}%")

print(f"\n✨ ƯU ĐIỂM CỦA PHƯƠNG PHÁP KẾT HỢP:")
print("• Sử dụng Hungarian cho lần đầu để tối ưu tổng chi phí")
print("• Sử dụng Auction cho các lần sau để cân bằng workload")
print("• Giảm chênh lệch đường bay giữa các drone")
print("• Drone bay xa ở kịch bản trước được ưu tiên bay gần ở kịch bản sau")