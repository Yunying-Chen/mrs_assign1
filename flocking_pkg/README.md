# MRS_Assignment 1

## Files
1. flocking_node.py : subscribe and publish topic to each boid
2. flocking.py: flocking algorithm 

## Usage
1. Create a package name `flocking_pkg` under `src` folder 
2. Put the files under this package and modified the `setup.py`
3. Do `colcon build && source install/setup.bash` 
4. Run simluation and `ros2 run flocking_pkg flocking_node`

## ToDo List
- [x] Subscribe and Publish topic for each boid
- [x] Alignment
- [x] Cohesion
- [x] Seperation
- [x] Subscribe for the map
- [x] Obstacle Avoidance 
- [x] Navigation to Point 
- [ ] fine tuning
- [ ] Test in Map a 
- [ ] Test in Map b 
- [ ] Test in Map c
- [ ] Test in Map d

## Updates
- Changed `flocking_node.py` to read the number of the robot from yml and simplified the subs and pubs
- Added the navigation to waypoints in `flocking.py`

