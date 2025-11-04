#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy                           
from rclpy.node   import Node      
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from flocking_pkg.flocking import *
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory
import yaml
import os
class FlockingNode(Node):
    def __init__(self,name):
        super().__init__(name)
        # Subscriptions to odometry topics 
        config_path = '/root/ros2_ws/src/flocking_pkg/flocking_pkg/params.yaml' 
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.declare_parameter('num_of_robots', 6)               
        self.num_of_robots = self.get_parameter('num_of_robots').get_parameter_value().integer_value
        self.get_logger().info(f'Starting flocking with {self.num_of_robots} robots')


        self.subs = {}
        self.pubs = {}
        self.flocking = None
        self.boids_dict = {}
        # self.waypoints = [(-4,2),(4,4),(-4,-4),(4,-2)]
        self.waypoints = self.params["waypoints"]
        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self.map_sub, qos)
        self.leader_id = self.params["leader_id"]
        self.leaderWeight = self.params["leaderWeight"]
        self.followLeader = self.params["followLeader"]

        
        self.timer = self.create_timer(0.5, self.flocking_callback)
        self.robots_setup()
        

    def flocking_callback(self):
        if self.flocking is None or len(self.boids_dict) < 2:
            return
        updated_boids = self.flocking.update_boids(dt=0.5)
        # 
        for boid_id, boid in updated_boids.items():
            # self.get_logger().info(f'Boid {boid_id} velocity: {boid.velocity}')
            cmd_msg = Twist()
            cmd_msg.linear.x = boid.velocity[0]
            cmd_msg.linear.y = boid.velocity[1]
            self.pubs[boid_id].publish(cmd_msg)


    def map_sub(self, map_data):
        self.get_logger().info(f'Received map data')
        resolution = map_data.info.resolution
        width = map_data.info.width
        height = map_data.info.height
        origin = map_data.info.origin.position
        origin = (origin.x, origin.y)
        resolution = resolution
        map_array = np.array(map_data.data).reshape(height, width)
        if self.flocking is None:
            self.flocking = Flocking(self.boids_dict,params=self.params, map_array=map_array,resolution=resolution, origin=origin)
            if self.followLeader:
                self.flocking.set_leader(self.leader_id,self.leaderWeight)
            if len(self.waypoints) >0:
                goalWeight = self.params["goalWeight"]
                self.flocking.set_waypoints(self.waypoints,goalWeight=goalWeight)

    def robots_setup(self):
        for sub in self.subs.values():
            self.destroy_subscription(sub)
        for pub in self.pubs.values():
            self.destroy_publisher(pub)
        self.subs.clear()
        self.pubs.clear()

        for i in range(self.num_of_robots):
            self.subs[i] = self.create_subscription(Odometry, f"/robot_{i}/odom", self.robot_odom_callback(i), 10)
            self.pubs[i] = self.create_publisher(Twist, f"/robot_{i}/cmd_vel", 10)


    def robot_odom_callback(self, robot_id):
        def callback(odom_data):
            curr_time = odom_data.header.stamp
            pose = odom_data.pose.pose.position
            linear_vel = odom_data.twist.twist.linear
            boid = Boid(robot_id, (linear_vel.x, linear_vel.y), (pose.x, pose.y))
            self.boids_dict[robot_id] = boid
        return callback

def main(args=None):
    rclpy.init(args=args)
    node = FlockingNode("flocking_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()