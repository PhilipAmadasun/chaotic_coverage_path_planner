# Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning

# Authors
* Uyiosa Philip Amadasun
* Patrick McNamee
* Zahra Nili Ahmadabadi
* Peiman Naseradinmousavi

## Project Description
This project is the code portion of an [arXiv submission](https://arxiv.org/abs/2305.01834) for the paper "Autonomous Search of Real-Life Environments Combining Dynamical System-Based Path Planning and Unsupervised Learning". This README file provides instructions for installing the neccessary software to build and run the codes. The platforms used to develop these codes are Robot Operating System (ROS). The codes are developed to run in both ROS kinetic and ROS melodic. The following will provide instructions to build and run the chaotic_cpp package with the Turtlebot2 and Turtlebot3, the robots used in the published work. If using this code repository, please cite the work as follows.

```
@misc{amadasun2023autonomous,
      title={Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning}, 
      author={Uyiosa Philip Amadasun and Patrick McNamee and Zahra Nili Ahmadabadi and Peiman Naseradinmousavi},
      year={2023},
      eprint={2305.01834},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Prerequisites

- Ubuntu 16.04 LTS for [ROS kinectic](http://wiki.ros.org/kinetic) 
- Ubuntu 18.04 LTS for [ROS melodic](http://wiki.ros.org/melodic)
- A working network connection

## Turtlebot3 and Turtlebot2 packages
* For instructions on installing and using Turtlebot3 ROS packages see [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)

* For instructions on installing and using Turtlebot2 ROS packages see [here](https://learn.turtlebot.com/)

* Make sure the any of the various exports needed to use these packages are in your bashrc file or are loaded in  your terminal

## Building the chaotic_cpp package
1. Create catkin workspace if you have not.
  ```
  mkdir -p ~/catkin_ws/src
  ```
 
2. Move into the source directory of your catkin workspace and clone this package into a directory named `chaotic_cpp` 
  ```
  cd ~/catkin_ws/src && git clone  git@gitlab.com:dsim-lab/paper-codes/Autonomous_search_of_real-life_environments.git chaotic_cpp 
  ```

3. Build the package
  ```
  catkin build #OR cd .. && catkin_make
  ```

4. Make sure to source your workspace to your bashrc file.

## Instructions to test chaotic_cpp package in ROS (kinetic and melodic)

### Using package with simulated Turtlebot3

1. Assuming all the necessary packages required to simulate a [Turtlebot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/#gazebo-simulation) are installed, open up a simulation of a Turtlebot3 in a gazebo environment with a known map (via SLAM) using the launch file shown:

    ```bash
    roslaunch turtlebot3_gazebo turtlebot3_world.launch
    ```

2. In another terminal, open the map in RVIZ

    ```bash
    roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=<your_map.yaml>
    ```

3. Make sure your robot is localized on the map using the `2D_Pose_Estimate` buttons on your RVIZ GUI (see [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/#run-navigation-nodes))


4. Now launch the chaotic coverage path planner

    ```bash
    roslaunch chaotic_CPP CCPP.launch
    ```
  * Make sure workspace is sourced

### Using package with real-life Turtlebot3

1. Make sure to  connect Turtlebot to remote PC and  bringup the drivers

  ```bash
    ssh pi@{IP_ADDRESS_OF_RASPBERRY_PI}
    roslaunch turtlebot3_bringup turtlebot3_robot.launch
  ```

2. Make sure you have a map of the environment you want to create cover.
  You can use SLAM to generate the map

    ```bash
    roslaunch turtlebot3_slam turtlebot3_slam.launch
    ```

  Save the map by running `rosrun map_server map_saver -f ~/map`

3. Now load the map and the robot model to RVIZ
    ```bash
    roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=<your_map.yaml>
    ```
4. Make sure your robot is localized on the map using the `2D_Pose_Estimate` buttons on your RVIZ GUI (see [here](https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/#run-navigation-nodes))

5. Run the chaotic coverage path planner like discussed previously.

### Using package with real-life Turtlebot2

1. To bring up the Turtlebot2 base, use the `turtlebot_bringup` package:

    ```bash
    roslaunch turtlebot_bringup minimal.launch
    ```


2. For running the navigation stack (ensure you have a map of the environment):

    ```bash
    roslaunch turtlebot_navigation amcl_demo.launch map_file:=<your_map.yaml>
    ```
  * If you don't have a map create one:
      ```bash
      roslaunch turtlebot_navigation gmapping_demo.launch
      ```
3. To visualize the robot and its sensors in RViz, run:

    ```bash
    roslaunch turtlebot_rviz_launchers view_robot.launch
    ```
4. Make sure your robot is localized on the map using the `2D_Pose_Estimate` buttons on your RVIZ GUI just as you do for the Turtlebot3.

5.  Run the chaotic coverage path planner like discussed previously.

# Paper Correspondence
  The rest of the README goes over correspondance between paper and codebase and gives descriptions of the code files.

# Algorithms
All codes in `src` directory.
* Algorithm 1 (ArnoldTrajectoryPlanner): 
Corresponds to `ArnoldLogistic_coverage` class method in `arnold_logistic_coverage.py`.
* Algorithm 2 (Shift):  Corresponds to `shift` class method in `arnold_logistic_coverage.py`.
* Algorithm 3 (CostCalculator): Corresponds to `cost_calculator` class method in `arnold_logistic_coverage.py`.
* Algorithm 4 (CoverageCalculator): Corresponds to `logistician` class method (which calls the `Coverage_Calculator` class method) in `map_zoner.py`. The `logistician` continuously calls the `Coverage_Calculator` in a loop. The  `logistician` continuously provides new sensor data to the `Coverage_Calculator`.
* Algorithm 5 (Worker):  Corresponds to `worker` class method in `map_zoner.py`.

Chaos control Techniques
1. Obstacle Avoidance technique
    * Quadtree algorithms imported from `quadtree.py`  are used in the `mapmaker` class method in`arnold_logistic_coverage.py` to create quadtree and query from it using `shift` class method.
    * DS_index switching by changing dynamical system variable index as seen in `arnold_logistic_coverage.py`.
2. Map-zoning
    * The `mapmaker` class method of `map_zoner.py` divides the map into zones and records the zone centroids in `self.m_z`. The Kmeans clustering used is imported from the Python 2.7 `sklearn` module.
    * The centroid of least covered zones is assigned to `self.x_y_zones` and is published via ROS in `map_zoner.py`. The ArnoldTrajectoryPlanner receives the zone centroid through a subscriber.

Real-time computation technique for coverage calculation
1. Quadtree algorithms imported from `quadtree.py`  are used in the `mapmaker` class method in`map_zoner.py` to create quadtree and query from it using `Coverage_Calculator` class method.
2. Cell coverage status update is stored in `self.m_c` which is created by the `tablemaker` class method in `map_zoner.py`.
3. The mutithreading option is set up in the `Coverage_Calculator` class method in `map_zoner.py`.

# Files

## config

### Description 
Contains all the configuration files used to set parameters of the chaotic coverage path planner on launch via `CCPP.launch` file.

ArnoldLogistic.yaml
* dt - time step
* n_iter - a total number of iterations
* ns - a set of iterations
* dist_to_goal - threshold of number of subgoals to trajectory point before switching to next trajectory point 
* v - robot velocity
* A - Arnold system parameter
* B - Arnold system parameter
* C - Arnold system parameter

mapzoner.yaml
* zones - number of zones to break the free space in the environment into.
* sensing_range - querying radius (meters) for coverage calculation
* NumberofThreads - number of threads of Algorithm 5 to be created.
* scan_frame - name given to the sensor frame of the robot.
* Multi-threading - optionality for multi-threading. `1` to enable, `0` to disable.
* desired coverage - desired coverage rate

## launch

### Description 
Contains the launch files of application.

`CCPP.launch`
This file executes the `arnold_logistic_coverage.py` and `map_zoner.py` scripts as ROS nodes to accomplish the chaotic coverage path planning process.

`map_zoner.launch`
This file executes the `map_zoner.py` as a ROS node to accomplish coverage calculation.

## src

### Description 
Contains all the Python source code.

`quadtree.py` - imported in `arnold_logistic_coverage.py` and `map_zoner.py` to access the algorithms to create and query a quadtree.

`arnold_rk4.py` - imported by `arnold_logistic_coverage.py` to use RK4 method.

`arnold_logistic_coverage.py` - applies Algorithms 1 to 3.

`map_zoner.py` - applies Algorithms 4 to 5.
