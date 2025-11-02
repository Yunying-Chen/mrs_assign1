import numpy as np
import math
from scipy.spatial import cKDTree
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Boid:
    def __init__(self,id,velocity=np.zeros(2),position=np.zeros(2)):
        self.id = id
        self.velocity = velocity #(vx,vy)
        self.position = position #(x,y)
        self.heading = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) > 0 else 0.0
    
    def set_velocity(self,new_velocity):
        self.velocity = new_velocity
        if np.linalg.norm(new_velocity) > 0:
            self.heading = np.arctan2(new_velocity[1], new_velocity[0])
    
    def set_position(self,new_position):
        self.position = new_position

# empty map weights: weights=[0.4,0.3,0.3,0.4]
class Flocking:
    def __init__(self,boids,neighbor_distance=4,neighbor_fov=6.28, weights=[0.4, 0.3, 0.3, 0.4], \
                 world_size=10,map_array=None, resolution=1.0, origin=np.zeros(2), clearance=0.1):
        # boids info
        self.boids = boids
        self.num_boids = len(boids.keys())
        self.neighbor_distance =neighbor_distance
        self.neighbor_fov=neighbor_fov
        self.max_acc= 2.0
        self.max_vel= 0.3
        self.leader_id = 0 
        
        # weights for different force
        self.weights=weights
        self.seperationWeight=weights[0] 
        self.alignmentWeight=weights[1]
        self.cohesionWeight=weights[2]
        self.obstacleAvoidanceWeight=weights[3]
        self.goalWeight=0.0

        # map
        self.world_size=10
        self.map_array = map_array if map_array is not None else np.zeros((world_size, world_size)) 
        self.origin = origin
        self.resolution = resolution
        self.clearance = clearance

        # waypoints
        self.waypoints = None
        self.reached_goal = False  
       
        
    def set_leader(self, leader_id, leaderWeight=0.8):
        self.leader_id = leader_id
        self.leaderWeight = leaderWeight

    def set_waypoints(self, waypoints, goal_slowdown_radius=2.0, goal_stop_radius=0.5, goalWeight=1.0):
        self.waypoints = [np.array(p, dtype=float) for p in waypoints]
        self.goal_slowdown_radius = goal_slowdown_radius
        self.goal_stop_radius = goal_stop_radius
        self.goalWeight = goalWeight
        self.goal = waypoints[-1] 
        self.reached_goal = False

    def compute_neighbor_boids(self, current_boid_id):
        positions = np.array([boid.position for boid in self.boids.values()])
        boid_ids = list(self.boids.keys())
        tree = cKDTree(positions)
        
        current_boid = self.boids[current_boid_id]
        indices = tree.query_ball_point(current_boid.position, self.neighbor_distance)
        
        neighbors = []
        for idx in indices:
            neighbor_id = boid_ids[idx]
            if neighbor_id == current_boid_id:
                continue
            neighbor = self.boids[neighbor_id]

            direction = (neighbor.position[0] - current_boid.position[0], neighbor.position[1] - current_boid.position[1])
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue
            direction /= distance
            
            heading_vector = np.array([np.cos(current_boid.heading), np.sin(current_boid.heading)])
            cos_angle = np.dot(heading_vector, direction)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            if angle <= self.neighbor_fov / 2:
                neighbors.append(neighbor)
        
        return neighbors



    def compute_seperation(self,boid,neighbors):
        sep_vec = np.zeros(2)
        for neighbor in neighbors:
            direction = (neighbor.position[0] - boid.position[0], neighbor.position[1] - boid.position[1])
            distance = np.linalg.norm(direction)
            if 0 < distance < self.neighbor_distance:
                sep_vec += direction / (distance **2)
        norm = np.linalg.norm(sep_vec)
        # normalize seperation to unit vector
        return -sep_vec / norm if norm > 0 else np.zeros(2)

    def compute_cohesion(self,boid,neighbors):
        if len(neighbors)<1:
            return np.zeros(2)
        # compute the average position in neighbor
        center = np.mean([n.position for n in neighbors], axis=0)
        direction = center - boid.position
        norm = np.linalg.norm(direction)
        # normalize cohesion to unit vector
        return direction / norm if norm > 0 else np.zeros(2) 


    def compute_alignment(self,boid,neighbors):
        if len(neighbors)<1:
            return np.zeros(2)
        # compute the average neighbor velocity
        avg_velocity = np.mean([n.velocity for n in neighbors], axis=0)
        norm = np.linalg.norm(avg_velocity)
        # normalize alignment to unit vector
        return avg_velocity / norm if norm > 0 else np.zeros(2)

    def compute_obstacle_avoidance(self, boid):
        avoidance_vec = np.zeros(2)

        boid_grid_x = int((boid.position[0] - self.origin[0]) / self.resolution)
        boid_grid_y = int((boid.position[1] - self.origin[1]) / self.resolution)
        search_radius = int(self.neighbor_distance / self.resolution)
        # check if there is obstacle in the nearby grids
        for i in range(-search_radius, search_radius + 1):
            for j in range(-search_radius, search_radius + 1):
                grid_x = boid_grid_x + i
                grid_y = boid_grid_y + j
                if 0 <= grid_x < self.map_array.shape[0] and 0 <= grid_y < self.map_array.shape[1]:
                    if self.map_array[grid_x, grid_y] > 50:   
                        obstacle_pos = np.array([grid_x * self.resolution + self.origin[0] + self.resolution / 2,
                                                 grid_y * self.resolution + self.origin[1] + self.resolution / 2])
                        direction = boid.position - obstacle_pos
                        distance = np.linalg.norm(direction)
                        if distance < self.neighbor_distance and distance > 0:
                            force = 1.0 / (distance ** 2)
                            avoidance_vec += direction * force
        norm_total = np.linalg.norm(avoidance_vec)
        return avoidance_vec / norm_total if norm_total > 0 else np.zeros(2)

    def compute_steer_to_avoid(self, boid):
        steer_vec = np.zeros(2)
        tangent_points = []  # Collect candidate points

        # Boid's vision setup
        heading_vec = np.array([np.cos(boid.heading), np.sin(boid.heading)])
        pos = boid.position
        fov_half = self.neighbor_fov / 2
        max_dist = self.neighbor_distance

        # Grid search (same as before, but now for projection)
        grid_x = int((pos[0] - self.origin[0]) / self.resolution)
        grid_y = int((pos[1] - self.origin[1]) / self.resolution)
        search_r = int(max_dist / self.resolution) + 1

        for i in range(-search_r, search_r + 1):
            for j in range(-search_r, search_r + 1):
                gx, gy = grid_x + i, grid_y + j
                if 0 <= gx < self.map_array.shape[0] and 0 <= gy < self.map_array.shape[1]:
                    if self.map_array[gx, gy] > 50: 
                        obs_center = np.array([
                            gx * self.resolution + self.origin[0] + self.resolution / 2,
                            gy * self.resolution + self.origin[1] + self.resolution / 2
                        ])

                        obs_vec = obs_center - pos
                        obs_dist = np.linalg.norm(obs_vec)

                        if obs_dist > max_dist or obs_dist < 1e-6:
                            continue

                        proj_dot = np.dot(obs_vec / obs_dist, heading_vec)
                        if proj_dot <= 0:
                            continue  
                        enlarged_radius = self.resolution / 2 + self.clearance

                        obs_angle = np.arctan2(obs_vec[1], obs_vec[0])
                        tangent_angle = np.arcsin(enlarged_radius / obs_dist)

                        left_tangent_angle = obs_angle - tangent_angle
                        right_tangent_angle = obs_angle + tangent_angle
                        left_point = pos + obs_dist * np.array([
                            np.cos(left_tangent_angle),
                            np.sin(left_tangent_angle)
                        ])
                        right_point = pos + obs_dist * np.array([
                            np.cos(right_tangent_angle),
                            np.sin(right_tangent_angle)
                        ])

                        left_in_fov = abs(left_tangent_angle - boid.heading) <= fov_half
                        right_in_fov = abs(right_tangent_angle - boid.heading) <= fov_half

                        if left_in_fov:
                            tangent_points.append(left_point)
                        if right_in_fov:
                            tangent_points.append(right_point)

        if not tangent_points:
            return np.zeros(2)

        nearest_idx = 0
        min_angle = float('inf')
        for idx, point in enumerate(tangent_points):
            steer_dir = point - pos
            steer_dist = np.linalg.norm(steer_dir)
            if steer_dist < 1e-6:
                continue
            steer_unit = steer_dir / steer_dist
            angle_diff = abs(np.arctan2(steer_unit[1], steer_unit[0]) - boid.heading)
            if angle_diff < min_angle:
                min_angle = angle_diff
                nearest_idx = idx

        steer_to = tangent_points[nearest_idx] - pos
        steer_dist = np.linalg.norm(steer_to)
        if steer_dist < 1e-6:
            return np.zeros(2)

        return -steer_to / steer_dist
    
    def compute_obstacle_repulsion(self, boid, d_th=1.0):#, k=1.0):
        avoidance_vec = np.zeros(2)
        # boid coords
        boid_grid_x = int((boid.position[0] - self.origin[0]) / self.resolution)
        boid_grid_y = int((boid.position[1] - self.origin[1]) / self.resolution)
        search_window = int((self.neighbor_distance / self.resolution)/2) # half of search window
        # window search - detect border of obstacles
        min_x = max(0, boid_grid_x - search_window)
        max_x = min(self.map_array.shape[1],  boid_grid_x + search_window)
        min_y = max(0, boid_grid_y - search_window)
        max_y = min(self.map_array.shape[0], boid_grid_y + search_window)

        boid_scope = self.map_array[min_y:max_y, min_x:max_x]
        if boid_scope.size == 0: 
            print('No scope',boid_scope)
            return []

        obstacles_mask = boid_scope >= 80
        kernel = np.array([[1,1,1],
                            [1,0,1],
                            [1,1,1]])
        neighbors_count = convolve(obstacles_mask.astype(int), kernel, mode='constant', cval=0)
        borders_mask = obstacles_mask & (neighbors_count < 8) # occupied and at least 1 free neighbor cell
        # plt.imshow(borders_mask, cmap='grey')
        # plt.show()

        borders_idxs = np.argwhere(borders_mask)
        # to coordinates in real world (centered)
        for x_idx, y_idx in borders_idxs:
            x = self.origin[0] + (min_x + x_idx + 0.5)*self.resolution
            y = self.origin[1] + (min_y + y_idx + 0.5)*self.resolution
            obstacle_pos = np.array([x,y])
            direction = boid.position - obstacle_pos    # vector [2x1]
            distance = np.linalg.norm(direction)
            if distance <= d_th:
                # repulsion = k * ((1/distance)-(1/d_th))* (1/distance**2)
                repulsion = 1 - (distance/d_th)
                F = repulsion * (direction/distance)
                avoidance_vec += F
        norm = np.linalg.norm(avoidance_vec)
        return avoidance_vec / norm if norm > 0 else np.zeros(2)    # unit vector
        # return np.zeros(2)
        # return avoidance_vec if np.linalg.norm(avoidance_vec) > 0 else np.zeros(2)

    def navigate_to_waypoints(self, boid):
        if len(self.waypoints) == 0 or self.reached_goal:
            return np.zeros(2)

        current_goal = self.waypoints[0]
        direction = (current_goal[0] - boid.position[0], current_goal[1] - boid.position[1])
        dist = np.linalg.norm(direction)
        if dist < self.goal_stop_radius:
            self.waypoints.pop(0)
            if len(self.waypoints) == 0:
                self.reached_goal = True
                return np.zeros(2)  
            current_goal = self.waypoints[0]
            direction = (current_goal[0] - boid.position[0], current_goal[1] - boid.position[1])
            dist = np.linalg.norm(direction)
        unit = direction / dist if dist > 0 else np.zeros(2)
        slowdown_factor = np.clip(dist / self.goal_slowdown_radius, 0.0, 1.0) 
        attraction = unit * slowdown_factor
        if dist < self.goal_slowdown_radius:
            drag = -np.array(boid.velocity) * slowdown_factor * 3.0
            attraction += drag
        return attraction

    def compute_leader_attraction(self, boid):
        if boid.id == self.leader_id:
            return np.zeros(2)  
        leader_boid = self.boids.get(self.leader_id)
        direction = (leader_boid.position[0] - boid.position[0], leader_boid.position[1] - boid.position[1])
        norm = np.linalg.norm(direction)
        force = min(norm/1.5, 1.0) 
        return (direction / norm * force) if norm > 0 else np.zeros(2)
        

    @staticmethod
    def _add_prior_force(acc, desired, max_rem):
        if np.linalg.norm(desired) < 1e-9: return acc
        proj = np.dot(desired, acc)
        orth = desired - proj * acc / (np.linalg.norm(acc)**2 + 1e-12)
        room = max_rem - np.linalg.norm(acc)
        if room <= 0: return acc
        add = orth / np.linalg.norm(orth) * min(np.linalg.norm(orth), room)
        return acc + add

    def update_boids(self,dt):

        for i,boid in enumerate(self.boids.values()):

            ## using weighted sum of forces
            # neighbors = self.compute_neighbor_boids(boid.id)
            # alignment = self.compute_alignment(boid,neighbors) * self.alignmentWeight
            # cohesion = self.compute_cohesion(boid,neighbors) * self.cohesionWeight
            # seperation = self.compute_seperation(boid,neighbors) * self.seperationWeight
            # obstacle_avoidance = self.compute_obstacle_avoidance(boid) * self.obstacleAvoidanceWeight
            # leader_attraction = self.compute_leader_attraction(boid) * self.leaderWeight
            # goal_attraction=np.zeros(2)
            # if boid.id == self.leader_id:
            #     goal_attraction = self.navigate_to_waypoints(boid) * self.goalWeight
            # new_acc = alignment + cohesion + seperation + obstacle_avoidance + goal_attraction + leader_attraction

            ## using prioritized forces
            new_acc = np.zeros(2)
            neighbors = self.compute_neighbor_boids(boid.id)
            # obstacle_avoidance = self.compute_obstacle_avoidance(boid) 
            # obstacle_avoidance = self.compute_steer_to_avoid(boid)
            obstacle_avoidance = self.compute_obstacle_repulsion(boid)
            seperation = self.compute_seperation(boid,neighbors)
            goal_attraction = np.zeros(2)
            leader_attraction = np.zeros(2)
            if boid.id == self.leader_id:
                goal_attraction = self.navigate_to_waypoints(boid)
            else:
                leader_attraction = self.compute_leader_attraction(boid)   
            alignment = self.compute_alignment(boid,neighbors)
            cohesion = self.compute_cohesion(boid,neighbors)

            # Processing
            new_acc = self._add_prior_force(new_acc, goal_attraction, self.max_acc)
            remaining = self.max_acc - np.linalg.norm(new_acc)
            new_acc = self._add_prior_force(new_acc, leader_attraction, remaining)
            remaining = self.max_acc - np.linalg.norm(new_acc)  
            new_acc = self._add_prior_force(new_acc, seperation, remaining)
            remaining = self.max_acc - np.linalg.norm(new_acc)
            new_acc = self._add_prior_force(new_acc, obstacle_avoidance, remaining)
            remaining = self.max_acc - np.linalg.norm(new_acc)
            new_acc = self._add_prior_force(new_acc, cohesion, remaining)
            remaining = self.max_acc - np.linalg.norm(new_acc)
            new_acc = self._add_prior_force(new_acc, alignment, remaining)
            remaining = self.max_acc - np.linalg.norm(new_acc)


            
            norm_acc = np.linalg.norm(new_acc)
            if norm_acc > self.max_acc:
                new_acc = new_acc / norm_acc * self.max_acc
            new_vel = boid.velocity + new_acc * dt
            norm_vel = np.linalg.norm(new_vel)
            if norm_vel > self.max_vel:
                new_vel = new_vel / norm_vel * self.max_vel

            if self.reached_goal:
                if np.linalg.norm(np.array(boid.position) - np.array(self.boids[self.leader_id].position)) < self.goal_stop_radius:
                    new_vel = np.zeros(2)
            new_position = boid.position + new_vel * dt
            
            self.boids[i].set_velocity(new_vel)
            self.boids[i].set_position(new_position)
           
        return self.boids

