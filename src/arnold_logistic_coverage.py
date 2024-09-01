#!/usr/bin/python2.7

""" 
File: ARNOLD_LOGISTIC_COVERAGE.PY
Author: Uyiosa Philip Amadasun

Brief: 
        Responsible for the chaotic trajectory planning by inducing chaos in the non-linear dynamical Arnold system.
        This chaos is mapped unto the kinematics of a two-wheel differential drive robot. 
        Includes trajectory dispersal and obstacle avoidance decision making.

        classes included:
            Chaotic_system

References:


"""

import rospy
import math
from rospy.numpy_msg import numpy_msg
import numpy as np
import time
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from quadtree import Target, Rect, QuadTree
import tf
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from std_msgs.msg import Float64
import time
from arnold_rk4 import arnold_rk4


rospy.init_node("controlled_chaotic_trajectory_planner")

class Chaotic_system:
    """
    Chaotic_system class describes the chaotic trajectory panner.

    Attributes:
        start_stop (str): Empty string used to dictate the start and stop signals for a chaotic trajectory planner.
        pub (rospy.Publisher): Publisher object for publishing goals to the 'goal' topic.
        count (int): Counter initialized to 0, possibly used for keeping track of iterations or events.
        point (list): List to store points, initialized to an empty list.
        occ_gridmap (OccupancyGrid): Stores the occupancy grid map received from the '/map' topic.
        PDA (numpy.ndarray): Probability Data Array representing occupancy grid map information.
        free_cells (numpy.ndarray): Flattened array of index values for free cells in the occupancy grid.
        map_resolution (float): The resolution of the map from the occupancy grid.
        map_width (int): The width of the map from the occupancy grid.
        map_height (int): The height of the map from the occupancy grid.
        map_origin_x (float): The x coordinate of the map origin.
        map_origin_y (float): The y coordinate of the map origin.
        change (int): An integer attribute, initialized to 0, usage not specified.
        route (numpy.ndarray): An empty array to store route information.
        x_y_zone (list): A list to be assigned the centroids of zones for trajectory spreading.
        goal (PoseStamped): An object to store the goal pose.
        memory (int): An integer attribute, initialized to 0, usage not specified.
        x (int): An integer representing the x-coordinate, initialized to 0.
        y (int): An integer representing the y-coordinate, initialized to 1.
        z (int): An integer representing the z-coordinate, initialized to 0.
        IC_vec (list): List to contain initial conditions of a chaotic system.
        listener (tf.TransformListener): A tf class instance for transforming between frames.
        domain (Rect): Represents a 2D domain for quadtree structure.
        qtree (QuadTree): QuadTree object for spatial partitioning in 2D space.
        stage (int): Indicates the stage of the trajectory planning process 
        (1 for first stage, 2 for second, 3 for moving to least covered zone).
    """
    def __init__(self):
        self.start_stop = ""                                                      #Used in subscriber to topic that dicates 
                                                                                  #when to start and stop of 
                                                                                  #chaotic trajectory planner
        self.pub = rospy.Publisher("goal", PoseStamped, queue_size=1, latch=True)
        self.count = 0
        self.point = []
        self.occ_gridmap = rospy.wait_for_message("/map", OccupancyGrid)
        self.PDA = np.array(self.occ_gridmap.data)                                 #probability data array information 
                                                                                   #of occupancygrid map.

        self.free_cells = np.argwhere(self.PDA == 0).flatten()                     #index values of free cells 
                                                                                   #(from probability data array {PDA}).
        self.map_resolution = self.occ_gridmap.info.resolution
        self.map_width = self.occ_gridmap.info.width
        self.map_height = self.occ_gridmap.info.height
        self.map_origin_x = self.occ_gridmap.info.origin.position.x
        self.map_origin_y = self.occ_gridmap.info.origin.position.y
        self.change = 0
        self.route = np.array([])
        self.x_y_zone = []                                                          #This class attribute be assigned the centroids 
                                                                                    #of zones to spread trajectories to.
        self.goal = PoseStamped()                                                   # goal pose to describe trajectory point for ROS
        self.memory = 0
        self.x = 0
        self.y = 1
        self.z = 0

        self.IC_vec = []                                                            #List will contain initial conditions 
                                                                                    #of chaotic system.

        self.listener = tf.TransformListener()                                      #tf class function to find transformation
                                                                                    # Matrices between frames.

        self.domain = Rect((self.map_width) / 2, (self.map_height) / 2, 
                                        (self.map_width), (self.map_height)) 

        self.qtree = QuadTree(self.domain, 4)                                       #The 2D space which will 
                                                                                    #contain the quadtree struture.

        self.stage = 1  #Dictates if cost parameter "f" will be set to 0, depending on what part of the trajectory planning 
                        #process is being enacted. Value of 1, in "first stage" of Algorithm 1, Value of 2 in "second stage"
                        #of Algorithm 1. Value of 3 when robot must move to least covered zone.


    def call_back(self, msg):
        """
        Subscriber Callback function for controlling the start and stop of a chaotic trajectory planner.

        This method is called when a message is received. It updates the internal state to start or stop
        the chaotic trajectory planner based on the message data.

        Args:
            msg (std_msgs.msg.String): The incoming message data containing the command: "start" or "stop".
        """
        self.start_stop = msg.data

    def get_distance(self, first, second):
        """
        Calculate the Euclidean distance between two last trajectory point and cell.

        Args:
            first (list): The coordinates of the first point in the form [x, y].
            second (list): The coordinates of the second point in the form [x, y].

        Returns:
            float: The Euclidean distance between the two points.
        """
        distance = math.sqrt((second[1] - first[0]) ** 2 + (second[1] - first[1]) ** 2)
        return distance

    """
    These next 4 methods will provide a means of conversion of coordinates of cells
    """

    def x_y_M_to_cell_location(self, x_y_M): 
        """
        Convert a point coordinate from the map frame to a cell location in the occupancy-grid map.

        Args:
            x_y_M (list): A point coordinate in the map frame, represented as [x, y].

        Returns:
            list: A list containing the cell location in the occupancy-grid map, [cell_x, cell_y].
        """

        cell_x = math.ceil((x_y_M[0] - (self.map_origin_x)) / self.map_resolution)
        cell_y = math.ceil((x_y_M[1] - (self.map_origin_y)) / self.map_resolution)
        return [cell_x, cell_y]

    def cell_location_to_ind(self, cell_location): 
        """
        Convert a cell location in the occupancy-grid map to the corresponding cell index value in the PDA (probability data array).

        Args:
            cell_location (list): A list or tuple containing the cell location coordinates, [cell_x, cell_y].

        Returns:
            int: The index value in the PDA corresponding to the provided cell location.
        """
        return int((cell_location[1] * int(self.map_width)) + cell_location[0])

    def cell_location_to_x_y_M(self, cell_location):
        """
        Convert a cell location in the occupancy-grid map to a point coordinate in the map frame.

        Args:
            cell_location (list): A list or tuple containing the cell location coordinates, [cell_x, cell_y].

        Returns:
            list: A list containing the point coordinate in the map frame, [x, y].
        """
        x = (cell_location[0] * self.map_resolution) + self.map_origin_x
        y = (cell_location[1] * self.map_resolution) + self.map_origin_y
        return [x, y]

    def ind_to_cell_location(self, index):
        """
        Convert a cell index value in the PDA to the corresponding cell location in the occupancy-grid map.

        Args:
            index (int): The index value in the PDA (probability data array).

        Returns:
            list: A list containing the cell location in the occupancy-grid map, [cell_x, cell_y].
        """
        cell_x = index % self.map_width
        cell_y = math.ceil((index - cell_x) / self.map_width)
        return [cell_x, cell_y]

    def mapmaker(self):
        """
        Create a Quadtree from a list of cell locations representing free cells in the occupancy-grid map.

        This method creates a Quadtree data structure using the cell locations of free cells in the occupancy-grid map.
        """
        self.cells_x_y = []
        for free_cell in self.free_cells:
            self.cells_x_y.append(self.ind_to_cell_location(free_cell))

        self.points = [Target(*cell) for cell in self.cells_x_y]
        for point in self.points:
            self.qtree.insert(point)

    def choose_marker(self, msg): 
        """
        Subscriber callback to retrieve the coordinate of the least covered zone centroid.

        This method is a callback function for a subscriber. It receives a message containing the coordinate
        of the least covered zone centroid and updates the 'x_y_zone' attribute with this information.

        Args:
            msg (std_msgs.msg.Float64MultiArray): The incoming message data containing the centroid coordinate.
        """
        self.x_y_zone = [msg.data[0], msg.data[1]]

    def path_watcher(self, msg): 
        """
        Subscriber callback. Keeps track of the number of subgoals in the path plan to a trajectory point.

        This method is a callback function for a subscriber. It updates the 'route' attribute with
        the list of subgoals in the received path plan message.

        Args:
            msg (nav_msgs.msg.Path): The incoming message data containing the path plan with subgoals.
        """
        self.route = msg.poses

    """
    These next two functions are Algorithm 3 and 2 respectively
    They use the cost function (Cost = f + g) fo obstacle avoidance decision making.
    In this script the variables Cost_Total, fx and gx are the cost, f ang g respectively.
    variable l is the range for calculating parameter g. 
    The variable called radius is the quadtree querying range
    """
    def cost_calculator(self, ecol, erow):
        """ Algorithm 3
        Calculate the cost for obstacle avoidance decision making based on the cost function (Cost = f + g).

        This function calculates the cost for obstacle avoidance decision making using the cost function,
        where Cost = f + g. The function computes the cost for a specified range of cells around a given
        cell location.

        Args:
            ecol (int): The column index of the cell location.
            erow (int): The row index of the cell location.

        Returns:
            float: The calculated cost parameter g.
        """
        cell_bucket = []
        subset = []
        l = 6
        for i in range(1, l+1):
            icol = ecol - i
            col = ecol + i
            for j in range(0, l+1): #l
                row = erow + j
                irow = erow - j
                cell_bucket.append([col, row])
                cell_bucket.append([col, irow])
                cell_bucket.append([icol, row])
                cell_bucket.append([icol, irow])
        count = 0
        for cell in cell_bucket:
            subset.append(cell)
            try:
                if ecol>=self.map_width or erow>=self.map_height or ecol<=0 or erow<=0:
                    count += 500

                else:
                    count += abs(self.PDA[int((cell[1] * int(self.map_width)) + cell[0])])
            except:
                count += 500
        average = count / len(cell_bucket)
        return average

    def shift(self, n_TP_DS_R, cell_x, cell_y, prev_tp, radius):
        """ Algorithm 2
        This function calculates costs for different cell locations within a specified radius,
        considering both f and g parameters. 
        This function calls Algorithm 3 (method above)for this task. The function
        returns information about the cell location with the least cost.

        Args:
            n_TP_DS_R (list): The current trajectory point and corresponding DS coordinates.
            cell_x (int): The x-coordinate of the cell location.
            cell_y (int): The y-coordinate of the cell location.
            prev_tp (list): The previous trajectory point.
            radius (int): The querying range for the Quadtree.

        Returns:
            dict: A dictionary containing the next trajectory point (arnpnt) and its cost.
        """
        cost = 10e20  #Max possible cost
        query = []    
        if self.stage == 3:
            query.append([cell_x, cell_y])
        center, radius = [cell_x, cell_y], radius  # 7
        center = Target(*center)
        self.qtree.query_radius(center, radius,  query)

        if len(query) == 0:
            return {"arnpnt": n_TP_DS_R, "cost": 10e20}

        for point in query:
            self.point = self.cell_location_to_x_y_M(point)
            gx = self.cost_calculator(point[0], point[1])
            if self.stage == 1:
                fx = self.get_distance(self.point, prev_tp)
            elif self.stage == 2 or self.stage == 3:
                fx = 0
            Cost_Total = fx + gx
            if Cost_Total==0:
                cost = Cost_Total
                n_TP_DS_R[3] = self.point[0]
                n_TP_DS_R[4] = self.point[1]
                break
            if Cost_Total < cost:
                cost = Cost_Total
                n_TP_DS_R[3] = self.point[0]
                n_TP_DS_R[4] = self.point[1]

        return {"arnpnt": n_TP_DS_R, "cost": cost}


    def ArnoldLogistic_coverage(self, A, B, C, v):
        """
        This class method contains Algorithm 1, it handles the chatotic trajectory planning
        and calls Algorithm 2 for obstacle avoidance decision making.

        Args:
            A (float):\
            B (float):---> Arnold system parameters
            C (float):/        
            v (float):     Robot velocity

        Raises:
            rospy.ROSException: If the ROS node has problems obtaining parameters from the server or waiting for a transform.
            rospy.ROSTimeMovedBackwardsException: If the time retrieved from the ROS system is 
                                                  earlier than a previously retrieved time,
                                                  indicating time has moved backwards.
            rospy.ROSInterruptException: If the sleep or wait operations are interrupted 
                                         by shutdown requests or similar interrupts.
            tf.LookupException: If there is no transformation available between requested frames.
            ValueError: If the input parameters A, B, C, or v are out of acceptable ranges or types.
            IndexError: If calculated cell indices are out of the range of the occupancy grid or probability distribution array.
            Exception:  If an unexpected or unhandled error occurs within the trajectory planning or shifting algorithms.
        """
        dt = rospy.get_param("dt")                      #time step
        n_iter = rospy.get_param("n_iter")              # Total number of iterations 
        ns = rospy.get_param("ns")                      # number of iterations
        dist_to_goal = rospy.get_param("dist_to_goal")  # number of subgoals on global plan to a trajectory point

        DS_ind = 2                                      # DS index.
        th1=69                                          # Cost Threshold for Algorithm 1's "first stage"
        th2=50                                          # Cost threshold for Algorithm 1's "second stage"

        #Initialize self.goal
        self.goal.header.stamp = rospy.get_rostime()    
        self.goal.header.frame_id = 'map'
        self.goal.pose.position.x = 0
        self.goal.pose.position.y = 0
        self.goal.pose.position.z = 0
        self.goal.pose.orientation.x = 0
        self.goal.pose.orientation.y = 0
        self.goal.pose.orientation.z = 0
        self.goal.pose.orientation.w = 1

        self.memory = 0                                 # Number of iterations.
                                                                                            
        self.set = ""                                   # Dictates if to discontinue current chaotic evolution
                                                        # Determine IC for the next iterations of trajectory points.
                                                        # Value of "" if to continue current evolution.
                                                        # Value of "NEW SET" if to discontinue and start a new one.

        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.5))
        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
        self.IC_vec = [self.x, self.y, self.z, trans[0], trans[1]]
        TP_DS_R = self.IC_vec # Temporary matrix of Arnold dynamical system and robot coordinates.

        
         
        bad_seed_count = 0 #These two variables help dictate what whether to move to least covered areas
        start = "new"      #if robot is unable to perform effecient coverage around it's immediate location

        while self.start_stop != "stop":
            if self.start_stop == "stop":
                continue

            """Algorithm 1"""
            # dictates if robot must move to a least covered zone
            if self.memory >= n_iter or bad_seed_count == 3:
                self.stage = 3
                start = "new"
                bad_seed_count=0
                self.set = "NEW SET"
                try:
                    cell_x_y_zone = self.x_y_M_to_cell_location(self.x_y_zone)
                    target = self.shift(n_TP_DS_R, cell_x_y_zone[0], cell_x_y_zone[1], [], 19)
                    n_TP_DS_R = target["arnpnt"]
                    self.goal.pose.position.x = n_TP_DS_R[3]
                    self.goal.pose.position.y = n_TP_DS_R[4]
                    self.pub.publish(self.goal)
                    rospy.sleep(0.5)
                    reach = len(self.route)
                    hlfway = reach / 2
                    ptick = time.time()

                    #These segments of code publish goals and dictate when to set new goals,
                    #based on information on the path plan from the /move_base/NavfnROS/plan topic.
                    while reach >= dist_to_goal:
                        if reach == 0:
                            break
                        self.pub.publish(self.goal)
                        rospy.sleep(0.5)
                        reach = len(self.route)
                        ptock = time.time()
                        pticktock = ptock - ptick
                        if pticktock > 550 and reach <= hlfway:
                            break
                        if reach <= 80:
                            tick = time.time()
                            while reach > dist_to_goal:
                                if reach == 0:
                                    break
                                self.pub.publish(self.goal)
                                rospy.sleep(0.5)
                                reach = len(self.route)
                                tock = time.time()
                                ticktock = tock - tick
                                if ticktock >= 110:
                                    break

                    self.memory = 0
                    viable_tp_count = 0 #Variable keeps track of viable trajectory points in a set of iterations.
                except:
                    pass
            if start=="start":
                try:
                    if point_ready == True:
                        TP_DS_R = [TP_DS_R[self.count - 1][0],   TP_DS_R[self.count - 1][1],   TP_DS_R[self.count - 1][2],
                                  self.goal.pose.position.x, self.goal.pose.position.y]
                    else:
                        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.01))
                        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                        cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                        target = self.shift([0,0,0,0,0], cell_x_y[0], cell_x_y[1], [], 10)
                        TP_DS_R = [self.x, self.y, self.z, target["arnpnt"][3], target["arnpnt"][4]]

                except:
                    self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))
                    (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                    cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                    target = self.shift([0, 0, 0, 0, 0], cell_x_y[0], cell_x_y[1], [], 10)
                    TP_DS_R = [self.x, self.y, self.z, target["arnpnt"][3], target["arnpnt"][4]]

            if self.set == "NEW SET":
                self.set = ""
                self.IC_vec = [self.x, self.y, self.z, self.goal.pose.position.x , self.goal.pose.position.y]
                TP_DS_R = self.IC_vec

            prev_tp = [TP_DS_R[3],  TP_DS_R[4]]
            Tp = []
            n_TP_DS_R = TP_DS_R

            start = "start"

            """First stage"""
            for i in range(0, ns-1):
                self.stage = 1
                arnpnt_0 = arnold_rk4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1], n_TP_DS_R[2], n_TP_DS_R[3], n_TP_DS_R[4], dt, DS_ind)
                arnpnt = arnpnt_0
                index1 = DS_ind
                cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                ind = self.cell_location_to_ind([cell_x, cell_y])
                if cell_x >= self.map_width or cell_y >= self.map_height or  cell_x  <= 0 or cell_y <= 0:
                    n_TP_DS_R = arnpnt
                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                    prev_tp = [n_TP_DS_R[3], n_TP_DS_R[4]]
                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                    break

                if self.PDA[ind] == 0:
                    n_TP_DS_R = arnpnt
                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])

                elif  self.PDA[ind] != 0:
                    target1 = self.shift(arnpnt, cell_x, cell_y, prev_tp, 19)
                    if target1["cost"] >= th1: 
                        for i in range(0, 3):
                            if i != index1:
                                index2 = i
                                break
                        arnpnt_1 = arnold_rk4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1], n_TP_DS_R[2],n_TP_DS_R[3],n_TP_DS_R[4], dt, index2)
                        arnpnt = arnpnt_1
                        cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                        cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                        ind = self.cell_location_to_ind([cell_x, cell_y])

                        if cell_x >= self.map_width or cell_y >= self.map_height or cell_x <= 0 or cell_y <= 0:
                            n_TP_DS_R = arnpnt_0 
                            Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                            TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                            break
                        if self.PDA[ind] == 0:
                            n_TP_DS_R = arnpnt
                            Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                            TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                        elif self.PDA[ind] != 0:
                            target2 = self.shift(arnpnt, cell_x, cell_y, prev_tp, 19)  
                            if target2["cost"] >= th1: 
                                for i in range(0, 3):
                                    if i != index2 and i != index1:
                                        index3 = i
                                        break
                                arnpnt = arnold_rk4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1],n_TP_DS_R[2],n_TP_DS_R[3],n_TP_DS_R[4],dt, index3)
                                cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                                cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                                ind = self.cell_location_to_ind([cell_x,  cell_y])
                                if cell_x>=self.map_width or  cell_y>=self.map_height or cell_x<=0 or  cell_y<=0:
                                    n_TP_DS_R = arnpnt_1 
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    break
                                if self.PDA[ind] == 0:
                                    n_TP_DS_R = arnpnt
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                elif self.PDA[ind] != 0:
                                    minimum = min([target1["cost"], target2["cost"]])
                                    if minimum == target1["cost"]:
                                        n_TP_DS_R = target1["arnpnt"]
                                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    else:
                                        n_TP_DS_R = target2["arnpnt"]
                                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            else:
                                n_TP_DS_R = target2["arnpnt"]
                                Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                    else:
                        n_TP_DS_R = target1["arnpnt"]
                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                        Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])

            self.count = 0
            viable_tp_count = 0

            """Second stage"""
            while self.count < len(Tp):
                self.stage = 2
                no_shift=1
                if self.start_stop == "stop":
                    break

                cell_x = math.ceil((Tp[self.count][0] - (self.map_origin_x)) / self.map_resolution)
                cell_y = math.ceil((Tp[self.count][1] - (self.map_origin_y)) / self.map_resolution)


                if cell_x >= self.map_width or cell_y >= self.map_height or cell_x <= 0 or cell_y <= 0:
                    point_ready = False
                    self.count = self.count + 1
                    continue


                try:
                    if self.cost_calculator(cell_x, cell_y) >= th2 or  self.PDA[int((cell_y * self.map_width) + cell_x)] == -1 or  self.PDA[int((cell_y * self.map_width) + cell_x)] == 100:
                        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0),rospy.Duration(0.01))
                        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                        cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                        target = self.shift([0,0,0,0,0], cell_x_y[0], cell_x_y[1], [], 10)

                        if target["cost"] == 10e20:
                            point_ready = False
                        else:
                            Tp[self.count][0] = target["arnpnt"][3]
                            Tp[self.count][1] = target["arnpnt"][4]
                            self.goal.pose.position.x = Tp[self.count][0]
                            self.goal.pose.position.y = Tp[self.count][1]
                            point_ready = True
                    else:
                        point_ready = True
                        self.goal.pose.position.x = Tp[self.count][0]
                        self.goal.pose.position.y = Tp[self.count][1]
                    no_shift=0

                except IndexError:
                    self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.01))
                    (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                    cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                    target = self.shift([0, 0, 0, 0, 0], cell_x_y[0], cell_x_y[1], [], 10)

                    if target["cost"] == 10e20:
                        point_ready = False
                    else:
                        Tp[self.count][0] = target["arnpnt"][3]
                        Tp[self.count][1] = target["arnpnt"][4]
                        self.goal.pose.position.x = Tp[self.count][0]
                        self.goal.pose.position.y = Tp[self.count][1]
                        point_ready = True
                    no_shift=0

                if no_shift:
                    point_ready = True
                    self.goal.pose.position.x = Tp[self.count][0]
                    self.goal.pose.position.y = Tp[self.count][1]

                if point_ready == False:
                    self.count += 1
                    continue

                if point_ready == True:
                    viable_tp_count += 1
                    self.count = self.count + 1
                    self.pub.publish(self.goal)
                    rospy.sleep(0.3)
                    reach = len(self.route)
                    hlfway = reach / 2
                    ptick = time.time()
                    while reach >= dist_to_goal:
                        if reach == 0:
                            break
                        self.pub.publish(self.goal)
                        rospy.sleep(0.3)
                        reach = len(self.route)
                        ptock = time.time()
                        pticktock = ptock - ptick
                        if pticktock > 550 and reach <= hlfway:
                            break
                        if reach <= 80:
                            tick = time.time()
                            while reach > dist_to_goal:
                                if reach == 0:
                                    break
                                self.pub.publish(self.goal)
                                rospy.sleep(0.3)
                                reach = len(self.route)
                                tock = time.time()
                                ticktock = tock - tick
                                if ticktock >= 110:
                                    break
                self.memory += self.count
                if viable_tp_count/(len(Tp)) <= 0.25:
                    bad_seed_count += 1

    def drive(self):
        rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.path_watcher, queue_size=1)
        rospy.Subscriber("/zones", Float64MultiArray, self.choose_marker, queue_size=1)

        v = rospy.get_param("v")
        A = rospy.get_param("A")
        B = rospy.get_param("B")
        C = rospy.get_param("C")

        while self.start_stop != "start":
            continue

        rospy.loginfo("Chaotic coverage path planner starting .....")
        self.ArnoldLogistic_coverage(A, B, C, v)

if __name__ == "__main__":
    try:
        arny = Chaotic_system()
        rospy.Subscriber("/startup_shutdown", String, arny.call_back)
        arny.mapmaker()
        arny.drive()

    except rospy.ROSInterruptException:
        pass




