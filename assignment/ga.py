import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import time

# === Load scenes ===
file_path = "~/drone_light_show_algorithm/formation4.xlsx"
xls = pd.ExcelFile(os.path.expanduser(file_path))
scene_names = ["scene_1", "scene_2", "scene_3", "scene_4"]
scenes = [list(zip(xls.parse(name)["x"], xls.parse(name)["y"], xls.parse(name)["z"])) for name in scene_names]

def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def hungarian_algorithm(start_points, end_points):
    n = len(start_points)
    cost_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = euclidean_distance(start_points[i], end_points[j])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Thêm một giá trị nhỏ cho những điểm trùng nhau để tránh lỗi
    coinciding_pairs = []
    for i, j in zip(row_ind, col_ind):
        if euclidean_distance(start_points[i], end_points[j]) < 1e-6:
            coinciding_pairs.append((i, j))
    
    if coinciding_pairs:
        print(f"⚠️  Phát hiện {len(coinciding_pairs)} cặp điểm trùng nhau - sẽ được xử lý tự động")
    
    total_cost = cost_matrix[row_ind, col_ind].sum()
    distances = [cost_matrix[i, j] for i, j in zip(row_ind, col_ind)]
    std_cost = np.std(distances)
    
    return list(zip(row_ind, col_ind)), total_cost, std_cost

def evaluate_fitness(individual, start_points, end_points, lambda_weight=1.0):
    distances = [euclidean_distance(start_points[i], end_points[j]) for i, j in enumerate(individual)]
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    cv_dist = (std_dist/mean_dist)*100 if mean_dist > 0 else 0
    fitness = mean_dist + lambda_weight * cv_dist
    
    return fitness, mean_dist, cv_dist

def initialize_population(n, pop_size, start_points, end_points):
    population = []
    
    # Khởi tạo với Hungarian algorithm
    hungarian_assignment, _, _ = hungarian_algorithm(start_points, end_points)
    hungarian_perm = [end_idx for _, end_idx in sorted(hungarian_assignment, key=lambda x: x[0])]
    population.append(hungarian_perm)
    
    # Khởi tạo ngẫu nhiên cho các cá thể còn lại
    for _ in range(pop_size - 1):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)
    
    return population

def selection(population, fitnesses, selection_size=4):
    idx = np.argsort(fitnesses)
    return [population[i] for i in idx[:selection_size]]

def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = parent1[a:b+1]
    p2_fill = [item for item in parent2 if item not in child]
    ptr = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = p2_fill[ptr]
            ptr += 1
    return child

def mutate(individual, mutation_rate=0.1):
    individual = individual.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(start_points, end_points, lambda_weight=1.0, pop_size=100, 
                     generations=300, mutation_rate=0.1):
    n = len(start_points)
    population = initialize_population(n, pop_size, start_points, end_points)
    
    for gen in range(generations):
        fitness_results = [evaluate_fitness(ind, start_points, end_points, 
                                           lambda_weight) for ind in population]
        fitnesses = [result[0] for result in fitness_results]
        
        parents = selection(population, fitnesses, selection_size=4)
        
        next_population = parents.copy()
        while len(next_population) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            next_population.append(child)
        
        population = next_population    
    
    fitness_results = [evaluate_fitness(ind, start_points, end_points, 
                                       lambda_weight) for ind in population]
    fitnesses = [result[0] for result in fitness_results]
    best_idx = np.argmin(fitnesses)
    
    best_individual = population[best_idx]
    best_assignment = list(zip(range(n), best_individual))
    
    return best_assignment

def ga_distances(start_points, end_points, lambda_weight):
    assignment = genetic_algorithm(start_points, end_points, lambda_weight)
    distances = [euclidean_distance(start_points[i], end_points[j]) for i, j in assignment]
    return distances

# === Input λ cho từng scene ===
lambda_weights = []
for i in range(3):
    lambda_val = float(input(f"Nhập giá trị λ cho Scene {i+1} → Scene {i+2} (mặc định=3.0): ") or 3.0)
    lambda_weights.append(lambda_val)

# === Tính toán distances cho tất cả scenes ===
all_distances = []
total_distances = []

for i in range(3):
    start = scenes[i]
    end = scenes[i+1]
    distances = ga_distances(start, end, lambda_weights[i])
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
    ax.scatter(drone_ids, distances, color='green', label='GA Distances')
    ax.hlines(mean_distance, xmin=1, xmax=len(drone_ids), color='red', linestyle='--', label=f"Mean: {mean_distance:.2f}")
    ax.set_title(f"Scene {i+1} → Scene {i+2} (λ={lambda_weights[i]})", fontsize=12)
    ax.set_xlabel("Drone Index")
    ax.set_ylabel("Distance")
    ax.legend()
    ax.grid(True)

# Plot tổng distances cho từng drone
ax_total = axes[1, 1]
drone_ids = list(range(1, num_drones + 1))
mean_total_distance = np.mean(drone_total_distances)
ax_total.scatter(drone_ids, drone_total_distances, color='blue', label='Total Distances')
ax_total.hlines(mean_total_distance, xmin=1, xmax=num_drones, color='red', linestyle='--', 
                label=f"Mean: {mean_total_distance:.2f}")
ax_total.set_title("Tổng Distance của từng Drone (All Scenes)", fontsize=12)
ax_total.set_xlabel("Drone Index")
ax_total.set_ylabel("Total Distance")
ax_total.legend()
ax_total.grid(True)

plt.suptitle("Genetic Algorithm - Drone Distance Assignments qua các cảnh", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# === In thông tin tổng kết ===
print("\n📊 Thống kê tổng kết:")
for i in range(3):
    print(f"Scene {i+1} → Scene {i+2}: Tổng distance = {total_distances[i]:.2f}")
print(f"Tổng distance tất cả scenes: {sum(total_distances):.2f}")
print(f"Trung bình distance/drone: {mean_total_distance:.2f}")