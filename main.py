import pygame
import numpy as np
import math
import csv
import os
import glob
import time
from numba import njit
import random

# config
FPS = 120 # all phyics run at 120 times per second !!
DT = 1.0 / FPS
POPULATION_SIZE = 150 # lower this if your hardware is lagging too much
MUTATION_RATE = 0.15 #default unless using dynamic mutation
NUM_SENSORS = 17 # you can customize this, this is the number of sensors that the car sees
INPUT_NODES = 18  # make this 1 + num_sensors. the extra input is speed
HIDDEN_NODES = 24 # higher you make this the "smarter" it gets, requires more training though
OUTPUT_NODES = 4  # dont change this (gas, brake, left, right)
TIMEOUT_SECONDS = 5.0  # 5 second timeout gets reset every checkpoint you cross
LEADERBOARD_CACHE = []

#    num_points,  radius
# Extreme    65,     400
# Insane     50,     350
# Hard       45,     350
# Normal     35,     300
# Easy       25,     500

# dont adjust noise and track_width

# PROCEDURAL TRACK GENERATOR # 35, 300, 230, 130
def generate_procedural_track(num_points=35, base_radius=300, noise=230, track_width=130):
    center_x, center_y = 700, 400
    angles = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    
    radii = [base_radius + random.uniform(-noise, noise) for _ in range(num_points)]
    
    for _ in range(4):
        smoothed = []
        for i in range(num_points):
            smoothed.append((radii[i-1] + radii[i]*2 + radii[(i+1)%num_points]) / 4)
        radii = smoothed
        
    center_pts = []
    for i in range(num_points):
        x = center_x + math.cos(angles[i]) * radii[i]
        y = center_y + math.sin(angles[i]) * radii[i]
        center_pts.append((x, y))
        
    out_p, in_p = [], []
    for i in range(num_points):
        p_prev = center_pts[i - 1]
        p_next = center_pts[(i + 1) % num_points]
        
        tx, ty = p_next[0] - p_prev[0], p_next[1] - p_prev[1]
        length = math.hypot(tx, ty)
        tx, ty = tx / (length + 0.001), ty / (length + 0.001)
        
        nx, ny = -ty, tx 
        
        out_p.append((center_pts[i][0] + nx * (track_width / 2), center_pts[i][1] + ny * (track_width / 2)))
        in_p.append((center_pts[i][0] - nx * (track_width / 2), center_pts[i][1] - ny * (track_width / 2)))
        
    walls, checkpoints, center_segs = [], [], []
    for i in range(num_points):
        nxt = (i + 1) % num_points
        walls.append((out_p[i][0], out_p[i][1], out_p[nxt][0], out_p[nxt][1]))
        walls.append((in_p[i][0], in_p[i][1], in_p[nxt][0], in_p[nxt][1]))
        
        checkpoints.append((out_p[i][0], out_p[i][1], in_p[i][0], in_p[i][1]))
        checkpoints.append((
            (out_p[i][0] + out_p[nxt][0])/2, (out_p[i][1] + out_p[nxt][1])/2,
            (in_p[i][0] + in_p[nxt][0])/2, (in_p[i][1] + in_p[nxt][1])/2
        ))
        
        center_segs.append((center_pts[i][0], center_pts[i][1], center_pts[nxt][0], center_pts[nxt][1]))
        
    start_pos = ((out_p[0][0]+in_p[0][0])/2, (out_p[0][1]+in_p[0][1])/2)
    dx = ((out_p[1][0]+in_p[1][0])/2) - start_pos[0]
    dy = ((out_p[1][1]+in_p[1][1])/2) - start_pos[1]
    start_angle = math.atan2(dy, dx)
    
    return np.array(walls, dtype=np.float32), np.array(checkpoints, dtype=np.float32), np.array(center_segs, dtype=np.float32), start_pos, start_angle

WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE = generate_procedural_track()

@njit(fastmath=True)
def calculate_physics_and_sensors(cx, cy, angle, sensor_angles, walls):
    distances = np.full(len(sensor_angles), 600.0)
    is_dead = False
    
    car_end_x = cx + math.cos(angle) * 15.0
    car_end_y = cy + math.sin(angle) * 15.0

    for i in range(len(walls)):
        wx1, wy1, wx2, wy2 = walls[i]
        
        num_c = (wy2-wy1)*(car_end_x-cx) - (wx2-wx1)*(car_end_y-cy)
        if num_c != 0:
            uA = ((wx2-wx1)*(cy-wy1) - (wy2-wy1)*(cx-wx1)) / num_c
            uB = ((car_end_x-cx)*(cy-wy1) - (car_end_y-cy)*(cx-wx1)) / num_c
            if 0 <= uA <= 1 and 0 <= uB <= 1:
                is_dead = True
        
        for j in range(len(sensor_angles)):
            ray_angle = angle + sensor_angles[j]
            rx = cx + math.cos(ray_angle) * 600.0
            ry = cy + math.sin(ray_angle) * 600.0
            
            num_r = (wy2-wy1)*(rx-cx) - (wx2-wx1)*(ry-cy)
            if num_r != 0:
                uA = ((wx2-wx1)*(cy-wy1) - (wy2-wy1)*(cx-wx1)) / num_r
                uB = ((rx-cx)*(cy-wy1) - (ry-cy)*(cx-wx1)) / num_r
                if 0 <= uA <= 1 and 0 <= uB <= 1:
                    dist = math.sqrt((uA*(rx-cx))**2 + (uA*(ry-cy))**2)
                    if dist < distances[j]:
                        distances[j] = dist
                        
    return distances, is_dead

@njit(fastmath=True)
def check_checkpoint(px, py, cx, cy, cp):
    cpx1, cpy1, cpx2, cpy2 = cp
    num = (cpy2-cpy1)*(cx-px) - (cpx2-cpx1)*(cy-py)
    if num != 0:
        uA = ((cpx2-cpx1)*(py-cpy1) - (cpy2-cpy1)*(px-cpx1)) / num
        uB = ((cx-px)*(py-cpy1) - (cy-py)*(px-cpx1)) / num
        if 0 <= uA <= 1 and -0.25 <= uB <= 1.25:
            return True
    return False

@njit(fastmath=True)
def dist_to_segment_array(px, py, segments):
    min_dist = 999999.0
    for i in range(len(segments)):
        x1, y1, x2, y2 = segments[i]
        l2 = (x2 - x1)**2 + (y2 - y1)**2
        if l2 == 0:
            dist = math.sqrt((px - x1)**2 + (py - y1)**2)
        else:
            t = max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2))
            proj_x = x1 + t * (x2 - x1)
            proj_y = y1 + t * (y2 - y1)
            dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist

class NeuralNetwork:
    def __init__(self, w1=None, w2=None):
        self.w1 = w1 if w1 is not None else np.random.randn(INPUT_NODES, HIDDEN_NODES)
        self.w2 = w2 if w2 is not None else np.random.randn(HIDDEN_NODES, OUTPUT_NODES)

    def forward(self, inputs):
        z1 = np.dot(inputs, self.w1)
        a1 = np.maximum(0, z1) 
        z2 = np.dot(a1, self.w2)
        z2 = np.clip(z2, -500, 500)
        return 1 / (1 + np.exp(-z2)) 

    def mutate(self, rate=MUTATION_RATE):
        new_w1 = self.w1 + np.random.randn(*self.w1.shape) * rate
        new_w2 = self.w2 + np.random.randn(*self.w2.shape) * rate
        return NeuralNetwork(new_w1, new_w2)

    @staticmethod
    def crossover(nn1, nn2):
        mask1 = np.random.rand(*nn1.w1.shape) > 0.5
        new_w1 = np.where(mask1, nn1.w1, nn2.w1)
        
        mask2 = np.random.rand(*nn2.w2.shape) > 0.5
        new_w2 = np.where(mask2, nn1.w2, nn2.w2)
        
        return NeuralNetwork(new_w1, new_w2)

class Car:
    def __init__(self, nn, name="AI", color=(255, 215, 0)):
        self.nn = nn
        self.name = name
        self.color = color
        
        self.engine_power = 900.0    
        self.brake_power = 3000.0    
        self.drag_coeff = 0.0045     
        self.rolling_resist = 100.0  
        self.wheelbase = 30.0
        self.lateral_grip = 600.0    
        self.max_wheel_angle = math.radians(35) 
        self.steering_speed = 2.5               
        
        self.sensor_angles = np.linspace(-math.pi/1.5, math.pi/1.5, NUM_SENSORS, dtype=np.float32)
        self.sensor_distances = np.full(NUM_SENSORS, 600.0, dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.x, self.y = START_POS
        self.angle = START_ANGLE 
        self.speed = 0.0
        self.current_wheel_angle = 0.0          
        self.is_braking = False      
        
        self.alive = True
        self.finished = False
        self.crashed = False
        
        self.fitness = 0.0
        self.checkpoints_passed = 0
        self.time_alive = 0.0
        self.time_since_checkpoint = 0.0
        self.lap_time = 0.0

    def update(self, use_center_reward=False):
        if not self.alive or self.finished: return
        self.time_alive += DT
        self.time_since_checkpoint += DT
        
        dists, dead = calculate_physics_and_sensors(self.x, self.y, self.angle, self.sensor_angles, WALLS)
        self.sensor_distances = dists
        
        if dead:
            self.alive = False
            self.crashed = True
            return
            
        if self.time_since_checkpoint > TIMEOUT_SECONDS:
            self.alive = False
            return 
        
        inputs = np.append(self.sensor_distances / 600.0, self.speed / 450.0)
        outputs = self.nn.forward(inputs)
        throttle, brake, left, right = outputs[0], outputs[1], outputs[2], outputs[3]
        
        self.is_braking = brake > 0.5
        
        force = (throttle * self.engine_power) - (brake * self.brake_power)
        force -= self.drag_coeff * (self.speed ** 2) 
        if self.speed > 0:
            force -= self.rolling_resist 
            
        self.speed += force * DT
        self.speed = max(0.0, self.speed) 
        
        target_wheel_angle = (right - left) * self.max_wheel_angle 
        angle_diff = target_wheel_angle - self.current_wheel_angle
        
        steer_step = self.steering_speed * DT
        if abs(angle_diff) < steer_step:
            self.current_wheel_angle = target_wheel_angle
        else:
            self.current_wheel_angle += math.copysign(steer_step, angle_diff)
            
        if self.speed > 10.0:
            min_turning_radius = (self.speed ** 2) / self.lateral_grip
            max_grip_angle = math.atan(self.wheelbase / min_turning_radius)
            effective_angle = max(-max_grip_angle, min(max_grip_angle, self.current_wheel_angle))
        else:
            effective_angle = self.current_wheel_angle
            
        if self.speed > 0.1 and abs(effective_angle) > 0.001:
            turning_radius = self.wheelbase / math.tan(effective_angle)
            angular_velocity = self.speed / turning_radius
            self.angle += angular_velocity * DT
        
        prev_x, prev_y = self.x, self.y
        self.x += math.cos(self.angle) * self.speed * DT
        self.y += math.sin(self.angle) * self.speed * DT
        
        base_fitness_gain = (self.speed * 0.01) * DT
        
        if use_center_reward:
            dist_to_center = dist_to_segment_array(self.x, self.y, CENTER_LINE)
            
            center_accuracy = max(0.0, 1.0 - (dist_to_center / 55.0)) 
            
            self.fitness += base_fitness_gain * (1.0 + center_accuracy * 3.0)
        else:
            self.fitness += base_fitness_gain 
        
        for offset in range(3):
            cp_idx = (self.checkpoints_passed + offset) % len(CHECKPOINTS)
            if check_checkpoint(prev_x, prev_y, self.x, self.y, CHECKPOINTS[cp_idx]):
                self.checkpoints_passed += (offset + 1)
                
                time_penalty = self.time_alive * 25.0
                self.fitness += max(500.0, 2000.0 - time_penalty)
                
                self.time_since_checkpoint = 0.0
                
                if self.checkpoints_passed >= len(CHECKPOINTS):
                    self.fitness += 10000.0 + (50000.0 / max(self.time_alive, 1.0))
                    self.finished = True
                    self.lap_time = self.time_alive
                    self.alive = False 
                break

def create_next_generation(population, generation, save_file, use_crossover=False, current_mutation_rate=0.15):
    population.sort(key=lambda x: x.fitness, reverse=True)
    best_car = population[0]
    
    np.savez(save_file, w1=best_car.nn.w1, w2=best_car.nn.w2)
    
    if best_car.finished:
        save_lap_time(generation, best_car.lap_time)
    
    next_gen = [Car(population[0].nn), Car(population[1].nn)] if len(population) > 1 else [Car(best_car.nn)]
    top_performers = population[:max(2, int(POPULATION_SIZE * 0.15))]
    
    while len(next_gen) < POPULATION_SIZE:
        if use_crossover and random.random() > 0.3:
            p1, p2 = np.random.choice(top_performers, 2, replace=False)
            child_nn = NeuralNetwork.crossover(p1.nn, p2.nn)
            next_gen.append(Car(child_nn.mutate(current_mutation_rate)))
        else:
            parent = np.random.choice(top_performers)
            next_gen.append(Car(parent.nn.mutate(current_mutation_rate)))
        
    return next_gen

def init_population(load_file=None):
    if load_file and os.path.exists(load_file):
        print(f"--> Booting existing model: {load_file}")
        data = np.load(load_file)
        base_nn = NeuralNetwork(data['w1'], data['w2'])
        return [Car(base_nn)] + [Car(base_nn.mutate()) for _ in range(POPULATION_SIZE - 1)]
    
    print("--> Booting brand new AI population.")
    return [Car(NeuralNetwork()) for _ in range(POPULATION_SIZE)]

def clear_leaderboard():
    global LEADERBOARD_CACHE
    LEADERBOARD_CACHE = []
    try:
        if os.path.exists('best_runs.csv'):
            os.remove('best_runs.csv')
    except Exception as e:
        pass

def save_lap_time(gen, time):
    filename = 'best_runs.csv'
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['Generation', 'Time'])
        writer.writerow([gen, round(time, 3)])
    update_leaderboard_cache()

def update_leaderboard_cache():
    global LEADERBOARD_CACHE
    if not os.path.isfile('best_runs.csv'): return
    runs = []
    with open('best_runs.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader: runs.append((int(row[0]), float(row[1])))
    runs.sort(key=lambda x: x[1]) 
    LEADERBOARD_CACHE = runs[:5]

def calculate_elo(rating_a, rating_b, score_a, k_factor=32):
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))
    
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b

# --- GAME MODES ---
def training_mode(screen, clock, font, load_model=None):
    global WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE
    camera_x, camera_y = -100, -100
    sim_speed, show_checkpoints, hyperspeed, generation = 1, False, False, 1
    auto_track_change = False 
    
    use_crossover = False
    use_dynamic_mutation = False
    use_center_reward = False
    
    base_mutation_rate = 0.25 
    current_mut_rate = MUTATION_RATE
    
    save_filename = load_model if load_model else "new_model.npz"
    population = init_population(load_model)
    update_leaderboard_cache() 
    running = True

    calculate_physics_and_sensors(0.0, 0.0, 0.0, population[0].sensor_angles, WALLS)
    focus_car = population[0]
    racing_line_trail = [] 

    print("\n" + "="*50)
    print(f" TRAINING INITIATED (Saving to: {save_filename})")
    print("="*50)
    gen_start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return False 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return True 
                if event.key == pygame.K_UP: sim_speed = min(500, sim_speed + 5) 
                if event.key == pygame.K_DOWN: sim_speed = max(1, sim_speed - 5)
                if event.key == pygame.K_c: show_checkpoints = not show_checkpoints
                if event.key == pygame.K_h: hyperspeed = not hyperspeed
                if event.key == pygame.K_x: use_crossover = not use_crossover
                if event.key == pygame.K_l: use_center_reward = not use_center_reward
                if event.key == pygame.K_m: 
                    use_dynamic_mutation = not use_dynamic_mutation
                    if use_dynamic_mutation: current_mut_rate = base_mutation_rate
                    else: current_mut_rate = MUTATION_RATE
                if event.key == pygame.K_r: 
                    WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE = generate_procedural_track()
                    population = init_population(save_filename)
                    clear_leaderboard()
                    focus_car = population[0]
                    racing_line_trail.clear()
                if event.key == pygame.K_t: 
                    auto_track_change = not auto_track_change
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: camera_y -= 15
        if keys[pygame.K_s]: camera_y += 15
        if keys[pygame.K_a]: camera_x -= 15
        if keys[pygame.K_d]: camera_x += 15

        steps = 100 if hyperspeed else sim_speed
        for step in range(steps):
            all_dead = True
            for car in population:
                if car.alive and not car.finished:
                    car.update(use_center_reward)
                    all_dead = False
            
            alive_cars = [c for c in population if c.alive or c.finished]
            if alive_cars:
                current_leader = max(alive_cars, key=lambda c: c.fitness)
                if not focus_car.alive or current_leader.fitness > focus_car.fitness + 100:
                    if focus_car != current_leader:
                        focus_car = current_leader
                        racing_line_trail.clear() 
                        
            if not hyperspeed and focus_car.alive and step == 0:
                racing_line_trail.append((focus_car.x, focus_car.y, focus_car.speed))
                if len(racing_line_trail) > 400: racing_line_trail.pop(0)
                    
            if all_dead:
                gen_end_time = time.time()
                gen_duration = gen_end_time - gen_start_time
                
                best_overall = max(population, key=lambda c: c.fitness)
                
                print(f"[Gen {generation:03d}] "
                      f"Fit: {best_overall.fitness:04.0f} | "
                      f"CPs: {best_overall.checkpoints_passed:02d}/{len(CHECKPOINTS)} | "
                      f"Time Alive: {best_overall.time_alive:4.1f}s | "
                      f"Mut Rate: {current_mut_rate:.3f}")
                
                population = create_next_generation(population, generation, save_filename, use_crossover, current_mut_rate)
                focus_car = population[0]
                generation += 1
                racing_line_trail.clear()
                gen_start_time = time.time() 
                
                if use_dynamic_mutation:
                    current_mut_rate = max(0.01, current_mut_rate * 0.98) 
                
                if auto_track_change and generation % 5 == 0:
                    print(f"Changing Track Automatically (Gen {generation})")
                    WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE = generate_procedural_track()
                    clear_leaderboard()
                break
            
        if not hyperspeed:
            screen.fill((25, 25, 25))
            
            for wx1, wy1, wx2, wy2 in WALLS:
                pygame.draw.line(screen, (200, 200, 200), (int(wx1-camera_x), int(wy1-camera_y)), (int(wx2-camera_x), int(wy2-camera_y)), 4)
            
            if show_checkpoints:
                for cx1, cy1, cx2, cy2 in CHECKPOINTS:
                    pygame.draw.line(screen, (0, 100, 0), (int(cx1-camera_x), int(cy1-camera_y)), (int(cx2-camera_x), int(cy2-camera_y)), 1)
                    
            if use_center_reward:
                for clx1, cly1, clx2, cly2 in CENTER_LINE:
                    pygame.draw.line(screen, (0, 150, 255), (int(clx1-camera_x), int(cly1-camera_y)), (int(clx2-camera_x), int(cly2-camera_y)), 2)

            if len(racing_line_trail) > 1:
                for i in range(1, len(racing_line_trail)):
                    px, py, pspeed = racing_line_trail[i-1]
                    cx, cy, cspeed = racing_line_trail[i]
                    r = max(0, min(255, 255 - int((cspeed / 440) * 255)))
                    g = max(0, min(255, int((cspeed / 440) * 255)))
                    pygame.draw.line(screen, (r, g, 0), (px-camera_x, py-camera_y), (cx-camera_x, cy-camera_y), 3)

            for car in population:
                if not car.alive and not car.finished: continue
                is_focus = (car == focus_car)
                color = (255, 215, 0) if is_focus else (50, 100, 200)
                cx, cy = int(car.x - camera_x), int(car.y - camera_y)
                pygame.draw.circle(screen, color, (cx, cy), 8)
                
                if car.is_braking:
                    bx, by = int(cx - math.cos(car.angle)*8), int(cy - math.sin(car.angle)*8)
                    pygame.draw.circle(screen, (255, 0, 0), (bx, by), 5)
                
                pygame.draw.line(screen, (255, 255, 255) if is_focus else (150, 150, 150), 
                                 (cx, cy), (int(cx + math.cos(car.angle)*15), int(cy + math.sin(car.angle)*15)), 2)
                
                if is_focus and show_checkpoints:
                    for i, angle in enumerate(car.sensor_angles):
                        d = car.sensor_distances[i]
                        ray_angle = car.angle + angle
                        rx, ry = cx + math.cos(ray_angle)*d, cy + math.sin(ray_angle)*d
                        pygame.draw.line(screen, (50, 50, 50), (cx, cy), (int(rx), int(ry)), 1)

            pygame.draw.rect(screen, (15, 15, 15), (1150, 0, 250, 800))
            pygame.draw.line(screen, (100, 100, 100), (1150, 0), (1150, 800), 2)
            
            y_offset = 15
            ui_texts = [
                "[TRAINING MODE]",
                f"File: {save_filename}",
                f"Generation: {generation}",
                f"Speed: {focus_car.speed:.0f} px/s",
                "",
                "___ GENETICS / REWARDS ___",
                f"L: Center Reward ({'ON' if use_center_reward else 'OFF'})",
                f"X: Crossover ({'ON' if use_crossover else 'OFF'})",
                f"M: Dynamic Mut. ({'ON' if use_dynamic_mutation else 'OFF'})",
                f"Cur Rate: {current_mut_rate:.3f}",
                "",
                "___ CONTROLS ___",
                "ESC: Main Menu",
                "WASD: Camera",
                f"UP/DOWN: Speed [{sim_speed}x]",
                "C: View Data | H: Hyperspeed",
                "R: Force New Track",
                f"T: Auto-Track ({'ON' if auto_track_change else 'OFF'})",
                "",
                "___ LEADERBOARD ___"
            ]
            
            for text in ui_texts:
                clr = (255, 215, 0) if "TRAINING MODE" in text else (200, 200, 200)
                if "ON" in text and ("Crossover" in text or "Dynamic" in text or "Auto-Track" in text or "Center" in text):
                    clr = (50, 255, 50)
                screen.blit(font.render(text, True, clr), (1165, y_offset))
                y_offset += 25
                
            for rank, (gen, l_time) in enumerate(LEADERBOARD_CACHE):
                screen.blit(font.render(f"{rank+1}. Gen {gen} - {l_time:.4f}s", True, (255, 215, 0)), (1165, y_offset))
                y_offset += 25

            pygame.display.flip()
            clock.tick(FPS) 
        else:
            pygame.event.pump()

def race_mode(screen, clock, font):
    global WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE
    camera_x, camera_y = -100, -100
    
    model_files = glob.glob("*.npz")
    if not model_files: return True 
        
    colors = [(255,50,50), (50,150,255), (50,255,50), (255,255,50), (255,50,255), (50,255,255), (255,150,50)]
    racers = []
    
    for i, file in enumerate(model_files):
        data = np.load(file)
        nn = NeuralNetwork(data['w1'], data['w2'])
        name = file.replace(".npz", "")
        c = Car(nn, name=name, color=colors[i % len(colors)])
        racers.append(c)

    calculate_physics_and_sensors(0.0, 0.0, 0.0, racers[0].sensor_angles, WALLS)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                return False 
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return True 
                if event.key == pygame.K_r: 
                    WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE = generate_procedural_track()
                    for r in racers: r.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: camera_y -= 15
        if keys[pygame.K_s]: camera_y += 15
        if keys[pygame.K_a]: camera_x -= 15
        if keys[pygame.K_d]: camera_x += 15

        for racer in racers: racer.update() # use_center_reward defaults to False here

        active_racers = [r for r in racers if r.alive or r.finished]
        if active_racers:
            active_racers.sort(key=lambda r: (r.checkpoints_passed, -r.time_alive), reverse=True)
            focus_car = active_racers[0]
        else:
            focus_car = racers[0]

        screen.fill((25, 25, 25))
        for wx1, wy1, wx2, wy2 in WALLS:
            pygame.draw.line(screen, (200, 200, 200), (int(wx1-camera_x), int(wy1-camera_y)), (int(wx2-camera_x), int(wy2-camera_y)), 4)
            
        for cx1, cy1, cx2, cy2 in CHECKPOINTS:
            pygame.draw.line(screen, (0, 50, 0), (int(cx1-camera_x), int(cy1-camera_y)), (int(cx2-camera_x), int(cy2-camera_y)), 1)

        for racer in racers:
            if not racer.alive and not racer.finished: continue
            cx, cy = int(racer.x - camera_x), int(racer.y - camera_y)
            pygame.draw.circle(screen, racer.color, (cx, cy), 8)
            
            if racer.is_braking:
                bx, by = int(cx - math.cos(racer.angle)*8), int(cy - math.sin(racer.angle)*8)
                pygame.draw.circle(screen, (255, 0, 0), (bx, by), 5)
            
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (int(cx + math.cos(racer.angle)*15), int(cy + math.sin(racer.angle)*15)), 2)
            
            name_surface = font.render(racer.name, True, racer.color)
            screen.blit(name_surface, (cx - name_surface.get_width()//2, cy - 25))

        pygame.draw.rect(screen, (15, 15, 15), (1150, 0, 250, 800))
        pygame.draw.line(screen, (100, 100, 100), (1150, 0), (1150, 800), 2)
        
        y_offset = 20
        screen.blit(font.render("[RACE MODE]", True, (50, 255, 50)), (1165, y_offset))
        y_offset += 30
        screen.blit(font.render("ESC: Menu  |  R: New Track", True, (200, 200, 200)), (1165, y_offset))
        y_offset += 40
        
        screen.blit(font.render("___ LIVE STANDINGS ___", True, (200, 200, 200)), (1165, y_offset))
        y_offset += 30
        
        def sort_key(r):
            if r.finished: return (2, -r.lap_time) 
            if r.crashed or not r.alive: return (0, r.checkpoints_passed) 
            return (1, r.checkpoints_passed) 

        leaderboard_racers = sorted(racers, key=sort_key, reverse=True)
        
        for rank, r in enumerate(leaderboard_racers):
            if r.finished: status = f"{r.lap_time:.4f}s"
            elif r.crashed: status = "CRASHED"
            elif not r.alive: status = "DNF"
            else: status = f"CP: {r.checkpoints_passed}"
            
            text = f"{rank+1}. {r.name[:40]} - {status}"
            screen.blit(font.render(text, True, r.color), (1165, y_offset))
            y_offset += 25

        pygame.display.flip()
        clock.tick(FPS) 

def select_model_menu(screen, clock, font, title_font):
    while True:
        screen.fill((25, 25, 25))
        models = glob.glob("*.npz")
        
        title = title_font.render("TRAINING SETUP", True, (255, 255, 255))
        screen.blit(title, (1400//2 - title.get_width()//2, 100))
        
        y = 200
        opt0 = font.render("[0] Create brand new model", True, (50, 255, 50))
        screen.blit(opt0, (1400//2 - opt0.get_width()//2, y))
        y += 60
        
        for i, m in enumerate(models):
            if i >= 9: break 
            opt = font.render(f"[{i+1}] Continue Training: {m}", True, (200, 200, 200))
            screen.blit(opt, (1400//2 - opt.get_width()//2, y))
            y += 40
            
        y += 40
        cancel = font.render("Press ESC to return to Main Menu", True, (255, 100, 100))
        screen.blit(cancel, (1400//2 - cancel.get_width()//2, y))

        pygame.display.flip()
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "BACK"
                if event.key == pygame.K_0:
                    return None 
                for i in range(min(9, len(models))):
                    if event.key == getattr(pygame, f'K_{i+1}'):
                        return models[i]
                    
def calculate_los(wins, losses):
    if wins + losses == 0:
        return 50.0 
    z = (wins - losses) / math.sqrt(2 * (wins + losses))
    los = 0.5 + 0.5 * math.erf(z)
    return los * 100.0
                    
def tournament_mode(screen, clock, font):
    global WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE
    camera_x, camera_y = -100, -100
    
    model_files = glob.glob("*.npz")
    if len(model_files) < 2:
        return True 
        
    ratings = {file: 1200.0 for file in model_files}
    match_counts = {file: 0 for file in model_files}
    h2h_stats = {} 
    
    total_matches = 0
    running = True

    while running:
        p1_file, p2_file = random.sample(model_files, 2)
        matchup_key = tuple(sorted([p1_file, p2_file]))
        if matchup_key not in h2h_stats:
            h2h_stats[matchup_key] = [0, 0, 0] 
            
        is_p1_first = (p1_file == matchup_key[0])
        
        nn1 = NeuralNetwork(np.load(p1_file)['w1'], np.load(p1_file)['w2'])
        nn2 = NeuralNetwork(np.load(p2_file)['w1'], np.load(p2_file)['w2'])
        
        car1 = Car(nn1, name=p1_file.replace('.npz', ''), color=(50, 255, 50))
        car2 = Car(nn2, name=p2_file.replace('.npz', ''), color=(255, 50, 50))
        racers = [car1, car2]
        
        if total_matches % 5 == 0:
            WALLS, CHECKPOINTS, CENTER_LINE, START_POS, START_ANGLE = generate_procedural_track()
        
        for r in racers: r.reset()
        
        match_active = True
        while match_active and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return True
            
            for _ in range(100):
                all_done = True
                for r in racers:
                    if r.alive and not r.finished:
                        r.update()
                        all_done = False
                
                if all_done:
                    # Both finished - Fastest time wins
                    if car1.finished and car2.finished:
                        score_1 = 1.0 if car1.lap_time < car2.lap_time else 0.0
                    
                    # Only one finished - Finisher wins
                    elif car1.finished and not car2.finished:
                        score_1 = 1.0
                    elif not car1.finished and car2.finished:
                        score_1 = 0.0
                    
                    # NEITHER finished (Both DNF).... Use Checkpoints & Fitness
                    else:
                        if car1.checkpoints_passed > car2.checkpoints_passed:
                            score_1 = 1.0
                        elif car2.checkpoints_passed > car1.checkpoints_passed:
                            score_1 = 0.0 
                        else:
                            # Tied on Checkpoints? Tie-breaker: Who had better fitness (progress)?
                            if car1.fitness > car2.fitness:
                                score_1 = 1.0
                            elif car2.fitness > car1.fitness:
                                score_1 = 0.0
                            else:
                                score_1 = 0.5 # A very rare, true mathematical draw
                                
                    if score_1 == 1.0:
                        h2h_stats[matchup_key][0 if is_p1_first else 1] += 1
                    elif score_1 == 0.0:
                        h2h_stats[matchup_key][1 if is_p1_first else 0] += 1
                    else:
                        h2h_stats[matchup_key][2] += 1 
                        
                    new_r1, new_r2 = calculate_elo(ratings[p1_file], ratings[p2_file], score_1)
                    ratings[p2_file] = new_r2
                    match_counts[p1_file] += 1
                    match_counts[p2_file] += 1
                    total_matches += 1
                    
                    status_a = "Finished" if car1.finished else ("Crashed" if car1.crashed else "DNF")
                    status_b = "Finished" if car2.finished else ("Crashed" if car2.crashed else "DNF")
                    lap_a = round(car1.lap_time, 4) if car1.finished else ""
                    lap_b = round(car2.lap_time, 4) if car2.finished else ""
                    
                    log_filename = "tournament_match_log.csv"
                    file_exists = os.path.isfile(log_filename)
                    with open(log_filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['Model_A', 'Model_B', 'Score_A', 'Status_A', 'Status_B', 'Lap_A', 'Lap_B'])
                        writer.writerow([p1_file, p2_file, score_1, status_a, status_b, lap_a, lap_b])
                    
                    match_active = False
                    break

            screen.fill((25, 25, 25))
            for wx1, wy1, wx2, wy2 in WALLS:
                pygame.draw.line(screen, (100, 100, 100), (int(wx1-camera_x), int(wy1-camera_y)), (int(wx2-camera_x), int(wy2-camera_y)), 2)
                
            for racer in racers:
                if racer.alive or racer.finished:
                    cx, cy = int(racer.x - camera_x), int(racer.y - camera_y)
                    pygame.draw.circle(screen, racer.color, (cx, cy), 8)
                    
            pygame.draw.rect(screen, (15, 15, 15), (1150, 0, 250, 800))
            pygame.draw.line(screen, (100, 100, 100), (1150, 0), (1150, 800), 2)
            
            y_offset = 20
            screen.blit(font.render(f"[TOURNAMENT MODE]", True, (255, 215, 0)), (1165, y_offset))
            y_offset += 25
            screen.blit(font.render(f"Total Matches: {total_matches}", True, (200, 200, 200)), (1165, y_offset))
            y_offset += 40
            
            screen.blit(font.render("___ ELO RANKINGS ___", True, (200, 200, 200)), (1165, y_offset))
            y_offset += 30
            
            ranked = sorted(ratings.items(), key=lambda item: item[1], reverse=True)
            for rank, (file, elo) in enumerate(ranked):
                clean_name = file.replace('.npz', '')[:16]
                text = f"{rank+1}. {clean_name} - {int(elo)} ({match_counts[file]}m)"
                screen.blit(font.render(text, True, (255, 255, 255)), (1165, y_offset))
                y_offset += 25

            y_offset += 20
            screen.blit(font.render("___ TOP RIVALRIES (LOS) ___", True, (200, 200, 200)), (1165, y_offset))
            y_offset += 30
            
            active_matchups = sorted(h2h_stats.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
            
            for i in range(min(5, len(active_matchups))):
                key, stats = active_matchups[i]
                wA, wB, draws = stats
                
                if wA >= wB:
                    leader, trailer = key[0], key[1]
                    wins, losses = wA, wB
                else:
                    leader, trailer = key[1], key[0]
                    wins, losses = wB, wA
                    
                los_val = calculate_los(wins, losses)
                l_name = leader.replace('.npz', '')[:16]
                t_name = trailer.replace('.npz', '')[:16]
                text = f"{l_name} > {t_name} ({los_val:.1f}%) [{wins}-{losses}]"
                
                color = (50, 255, 50) if los_val >= 95.0 else (200, 200, 200)
                screen.blit(font.render(text, True, color), (1165, y_offset))
                y_offset += 25

            pygame.display.flip()
            clock.tick(FPS)

def main():
    pygame.init()
    screen = pygame.display.set_mode((1500, 800))
    pygame.display.set_caption("Formula Zero")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)
    title_font = pygame.font.SysFont("Arial", 40, bold=True)

    running = True
    while running:
        screen.fill((25, 25, 25))
        
        title = title_font.render("Formula Zero", True, (255, 255, 255))
        screen.blit(title, (1400//2 - title.get_width()//2, 200))
        
        opt1 = font.render("[1] Enter Training Mode", True, (200, 200, 200))
        opt2 = font.render("[2] Enter Race Mode", True, (200, 200, 200))
        opt3 = font.render("[3] Enter Tournament Mode", True, (255, 215, 0))
        
        screen.blit(opt1, (1400//2 - opt1.get_width()//2, 330))
        screen.blit(opt2, (1400//2 - opt2.get_width()//2, 380))
        screen.blit(opt3, (1400//2 - opt3.get_width()//2, 430))
        
        models = glob.glob("*.npz")
        if len(models) < 2:
            warn = font.render("Warning: Need at least 2 .npz files for a Tournament!", True, (255, 100, 100))
            screen.blit(warn, (1400//2 - warn.get_width()//2, 500))
        else:
            info = font.render(f"Found {len(models)} models ready for tournament.", True, (100, 255, 100))
            screen.blit(info, (1400//2 - info.get_width()//2, 500))

        pygame.display.flip()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selection = select_model_menu(screen, clock, font, title_font)
                    if selection == "QUIT": running = False
                    elif selection != "BACK": 
                        running = training_mode(screen, clock, font, load_model=selection)
                if event.key == pygame.K_2 and models:
                    running = race_mode(screen, clock, font)
                if event.key == pygame.K_3 and len(models) >= 2:
                    running = tournament_mode(screen, clock, font)

    pygame.quit()

if __name__ == "__main__":
    main()