import numpy as np
import math
from scipy.spatial import cKDTree
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


class Flocking:
    def __init__(self,boids,neighbor_distance=8,neighbor_fov=3.14,weights=[0.2,0.4,0.3,0.3], \
                 max_acc=1.0,max_vel=0.5, world_size=10,map_array=None, resolution=1.0, origin=np.zeros(2), clearance=0.1):
        self.boids = boids
        self.num_boids = len(boids.keys())
        self.neighbor_distance =neighbor_distance
        self.neighbor_fov=neighbor_fov
        self.weights=weights
        self.max_acc=max_acc
        self.max_vel=max_vel
        self.seperationWeight=weights[0] 
        self.alignmentWeight=weights[1]
        self.cohesionWeight=weights[2]
        self.obstacleAvoidanceWeight=weights[3]
        self.world_size=10
        self.map_array = map_array if map_array is not None else np.zeros((world_size, world_size))  # Default empty map
        self.grid_size_x = world_size / self.map_array.shape[0]
        self.grid_size_y = world_size / self.map_array.shape[1]
        self.origin = origin
        self.resolution = resolution
        self.clearance = clearance

    def compute_neighbor_boids(self, current_boid_id):

        # Build KD-tree from boid positions
        positions = np.array([boid.position for boid in self.boids.values()])
        tree = cKDTree(positions)
        
        current_boid = self.boids[current_boid_id]
        indices = tree.query_ball_point(current_boid.position, self.neighbor_distance)
        neighbors = []
        for idx in indices:
            if idx == current_boid_id:
                continue  
            neighbor = self.boids[idx]
            # Compute direction vector to neighbor
            direction = (neighbor.position[0] - current_boid.position[0], neighbor.position[1] - current_boid.position[1])
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue 
            # Normalize direction vector
            direction = direction / distance
            # Compute angle between heading and direction to neighbor
            heading_vector = np.array([np.cos(current_boid.heading), np.sin(current_boid.heading)])
            cos_angle = np.dot(heading_vector, direction)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            # Check if neighbor is within FOV
            if angle <= self.neighbor_fov / 2:
                neighbors.append(neighbor)
        
        return neighbors



    def compute_seperation(self,boid,neighbors):
        sep_vec = np.zeros(2)
        for neighbor in neighbors:
            direction = (neighbor.position[0] - boid.position[0], neighbor.position[1] - boid.position[1])
            distance = np.linalg.norm(direction)
            if 0 < distance < self.neighbor_distance:
                sep_vec += (neighbor.position[0] - boid.position[0], neighbor.position[1] - boid.position[1]) / distance
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
        # todo 
        pass



    def update_boids(self,dt):

        for i,boid in enumerate(self.boids.values()):

            neighbors = self.compute_neighbor_boids(boid.id)
            alignment = self.compute_alignment(boid,neighbors) * self.alignmentWeight
            cohesion = self.compute_cohesion(boid,neighbors) * self.cohesionWeight
            seperation = self.compute_seperation(boid,neighbors) * self.seperationWeight
            # obstacle_avoidance = self.compute_obstacle_avoidance(boid) * self.obstacleAvoidanceWeight

            new_acc = alignment + cohesion + seperation #+ obstacle_avoidance
            norm_acc = np.linalg.norm(new_acc)
            if norm_acc > self.max_acc:
                new_acc = new_acc / norm_acc * self.max_acc
            new_vel = boid.velocity + new_acc * dt
            norm_vel = np.linalg.norm(new_vel)
            if norm_vel > self.max_vel:
                new_vel = new_vel / norm_vel * self.max_vel

            new_position = boid.position + new_vel * dt

            self.boids[i].set_velocity(new_vel)
            self.boids[i].set_position(new_position)

        return self.boids



