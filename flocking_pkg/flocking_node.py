#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy                           
from rclpy.node   import Node      
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import Twist
from flocking_pkg.flocking import *

class FlockingNode(Node):
    def __init__(self,name):
        super().__init__(name)
        # Subscriptions to odometry topics 
        
        self.sub0 = self.create_subscription(Odometry, "/robot_0/odom", self.robot_0_sub, 10)
        self.sub1 = self.create_subscription(Odometry, "/robot_1/odom", self.robot_1_sub, 10)
        self.sub2 = self.create_subscription(Odometry, "/robot_2/odom", self.robot_2_sub, 10)
        self.sub3 = self.create_subscription(Odometry, "/robot_3/odom", self.robot_3_sub, 10)
        self.sub4 = self.create_subscription(Odometry, "/robot_4/odom", self.robot_4_sub, 10)
        self.sub5 = self.create_subscription(Odometry, "/robot_5/odom", self.robot_5_sub, 10)
        self.sub_map = self.create_subscription(OccupancyGrid, "/map", self.map_sub, 10)

        # Publisher to publish messages
        self.pub0 = self.create_publisher(Twist, "/robot_0/cmd_vel", 10)
        self.pub1 = self.create_publisher(Twist, "/robot_1/cmd_vel", 10)
        self.pub2 = self.create_publisher(Twist, "/robot_2/cmd_vel", 10)
        self.pub3 = self.create_publisher(Twist, "/robot_3/cmd_vel", 10)
        self.pub4 = self.create_publisher(Twist, "/robot_4/cmd_vel", 10)
        self.pub5 = self.create_publisher(Twist, "/robot_5/cmd_vel", 10)

        self.boids_dict = {}
        self.map=None
        self.origin = None
        self.resolution =None
        self.timer = self.create_timer(0.5, self.flocking_callback)

    def flocking_callback(self):
        # print(f'map status: {self.map is not None}, boids count: {len(self.boids_dict.keys())}')
        if len(self.boids_dict.keys())==6:
            flocking_controller = Flocking(self.boids_dict, map_array=self.map,resolution=self.resolution, origin=self.origin)
            updated_boids = flocking_controller.update_boids(dt=0.5)
            for boid_id, boid in updated_boids.items():
                cmd_msg = Twist()
                cmd_msg.linear.x = boid.velocity[0]
                cmd_msg.linear.y = boid.velocity[1]
                if boid_id == 0:
                    self.pub0.publish(cmd_msg)
                elif boid_id == 1:
                    self.pub1.publish(cmd_msg)
                elif boid_id == 2:
                    self.pub2.publish(cmd_msg)
                elif boid_id == 3:
                    self.pub3.publish(cmd_msg)
                elif boid_id == 4:
                    self.pub4.publish(cmd_msg)
                elif boid_id == 5:
                    self.pub5.publish(cmd_msg)


    def map_sub(self, map_data):
        print(f'Received map data')
        resolution = map_data.info.resolution
        width = map_data.info.width
        height = map_data.info.height
        origin = map_data.info.origin.position
        self.origin = origin
        self.resolution = resolution
        self.map = np.array(map_data.data).reshape(height, width)


    def robot_0_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_0 = Boid(0,(linear_vel.x,linear_vel.y),(pose.x, pose.y))        
        self.boids_dict[0] = boid_0

    def robot_1_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_1 = Boid(1,(linear_vel.x,linear_vel.y),(pose.x, pose.y))
        self.boids_dict[1] = boid_1
    
    def robot_2_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_2 = Boid(2,(linear_vel.x,linear_vel.y),(pose.x, pose.y))
        self.boids_dict[2] = boid_2
    
    def robot_3_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_3 = Boid(3,(linear_vel.x,linear_vel.y),(pose.x, pose.y))
        self.boids_dict[3] = boid_3
    
    def robot_4_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_4 = Boid(4,(linear_vel.x,linear_vel.y),(pose.x, pose.y))
        self.boids_dict[4] = boid_4

    def robot_5_sub(self, odom_data):
        curr_time = odom_data.header.stamp
        pose = odom_data.pose.pose.position
        linear_vel = odom_data.twist.twist.linear
        boid_5 = Boid(5,(linear_vel.x,linear_vel.y),(pose.x, pose.y))
        self.boids_dict[5] = boid_5

def main(args=None):
    rclpy.init(args=args)
    node = FlockingNode("flocking_node")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()