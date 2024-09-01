#!/usr/bin/python2.7
"""
File: MAP_ZONER.PY
Author: Uyiosa Philip Amadasun

Brief:
    Implements the map_zoner class to handle discritizing the occupancy map. When run as a
script, this file starts the `/zones` and `/coverage_rate` topics.

"""

import rospy
import math
import csv
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray,Int64MultiArray
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from std_msgs.msg import String
from quadtree import Target, Rect, QuadTree
import tf
import geometry_msgs.msg
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import LaserScan
import time
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

rospy.init_node("mapzoner")

class map_zoner:
    """
    Initializes the map_zoner object which is responsible for dividing an occupancy grid map into zones and 
    managing the coverage rate for each zone.

    Args:
        percentage (float): The initial coverage percentage, defaulted to 0.
        zones (int): The number of zones to divide the map into, retrieved from ROS parameter server.
        dc (float): The desired total coverage rate, retrieved from ROS parameter server.
        pub (rospy.Publisher): A publisher that sends out the centroid of the least covered zone as a 
                                Float64MultiArray message.
        pub2 (rospy.Publisher): A publisher for sending start-up and shut-down messages to activate or 
                                deactivate the chaotic coverage path planner.
        pub3 (rospy.Publisher): Another publisher for sending shut-down messages.
        pub4 (rospy.Publisher): A publisher that sends out the total coverage rate as a Float64 message.
        occ_gridmap (OccupancyGrid): A ROS message containing map metadata, including the map data which 
                                    is the occupancy probability of each cell.
        PDA (numpy.ndarray): An array representation of the occupancy grid map data.
        rows (numpy.ndarray): An array of indices representing the rows in the occupancy grid map.
        columns (int): The number of columns for an intermediate representation, defaulted to 5.
        m_c (numpy.ndarray): A matrix representing some intermediate data related to map cells.
        m_z (numpy.ndarray): A matrix representing zones and their attributes.
        free_cells (numpy.ndarray): The indices of free cells (cells not occupied or unknown) in the occupancy grid map.
        map_resolution (float): The resolution of the map (meters/cell).
        map_width (int): The width of the map in cells.
        map_height (int): The height of the map in cells.
        map_origin_x (float): The x-origin of the map in meters.
        map_origin_y (float): The y-origin of the map in meters.
        x_y_zones (Float64MultiArray): An array that will hold the zone centroid information after calculation.
        cells_x_y (list): A list that will be populated with the cell locations of free cells.
        points_x_y_M (list): A list that will be populated with the map coordinates of free cells.
        domain (Rect): A rectangle representing the domain in which the quadtree is defined.
        qtree (QuadTree): An instance of a quadtree data structure used to efficiently manage 2D spatial data.
        listener (tf.TransformListener): A TransformListener object to listen for transform updates.
        scan (LaserScan): A variable to hold laser scan data.
        finish (str): A string variable to record when the desired total coverage is reached.
        NOT (int): The number of threads for a specific algorithm, retrieved from ROS parameter server.
    """
    def __init__(self):
        self.percentage=0
        self.zones = rospy.get_param("zones")                                               #Number of zones.
        self.dc = rospy.get_param("desired coverage")                                       #Desired total coverage rate.

        self.pub = rospy.Publisher("/zones", Float64MultiArray, queue_size=1, latch=True)   #Publisher of zone centroid of
                                                                                            #least covered zone.

        self.pub2 = rospy.Publisher("/startup_shutdown", String, queue_size=1)              #startup and shutdown topics to
        self.pub3 = rospy.Publisher("/shutdown", String, queue_size=1)                      #communicate messages to start and 
                                                                                            #stop the chaotic coverage path planner. 
                                                                                            
        self.pub4 = rospy.Publisher("/coverage_rate", Float64, queue_size=1)                #Publisher of total coverage rate

        self.occ_gridmap = rospy.wait_for_message("/map", OccupancyGrid)                    #variables holds map metadata.
        self.PDA = np.array(self.occ_gridmap.data)
        self.rows = np.arange(len(self.PDA))
        self.columns = 5
        self.m_c = np.zeros((len(self.rows), self.columns))
        self.m_z = np.zeros((self.zones, 8))
        self.free_cells = np.argwhere(self.PDA == 0).flatten()                              #Indices of free cells.
        self.map_resolution = self.occ_gridmap.info.resolution
        self.map_width = self.occ_gridmap.info.width
        self.map_height = self.occ_gridmap.info.height
        self.map_origin_x = self.occ_gridmap.info.origin.position.x
        self.map_origin_y = self.occ_gridmap.info.origin.position.y

        self.x_y_zones = Float64MultiArray()                                                #Initialized variable that will
                                                                                            #contain zone centroid information.

        self.cells_x_y = []                                                                 #Will contain cell locations of free cells
        self.points_x_y_M = []                                                              #Will contain map coordinates o free cells

        self.domain = Rect((self.map_width) / 2, (self.map_height) / 2, 
                                        self.map_width, self.map_height)
        self.qtree = QuadTree(self.domain, 4)                                               #2D space that will contain quadtree.

        self.listener = tf.TransformListener()                                              #tf class method for retreiving 
                                                                                            #transform matrices.

        self.scan = LaserScan()                                                             #class variable used in
                                                                                            #subscriber callback,
                                                                                            #assigned laserscan data

        self.finish = " "                                                                   #class variable used in
                                                                                            #subscriber callback,
                                                                                            #assigned string data "finished" to record 
                                                                                            #when desired total coverage is reached.

        self.NOT = rospy.get_param("NumberofThreads")                                       #Number of threads of Algorithm 5

    def get_scan(self, msg):
        """
        Subscriber callback. Updates the scan attribute with the latest message received from a LaserScan ROS topic.

        Args:
            msg (LaserScan): The LaserScan message containing the latest scan data.
        """
        self.scan = msg

    def get_distance(self, center, point):
        """
        Calculate the Euclidean distance between the center (robot's location) and a free cell.

        Args:
            center (tuple): A tuple (x, y) representing the robot's current location.
            point (tuple): A tuple (x, y) representing the location of a free cell.

        Returns:
            float: The Euclidean distance between the robot's location and the free cell.
        """
        distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
        return distance

    def rad_to_deg(self, scan):
        """
        Converts scan angles from radians to degrees.

        Args:
            scan (LaserScan): A ROS LaserScan message.

        Returns:
            list: A list of scan angles converted from radians to degrees.
        """
        scan_angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment).tolist()
        for i in range(len(scan_angles)):
            scan_angles[i] = (int(math.degrees(scan_angles[i]))+360) % 360
        return scan_angles

    def transform(self, coord,mat44):
        """
        Transform points from the map frame to the scan frame for coverage calculation.

        Args:
            coord (list): A list [x, y] representing the coordinates in the map frame.
            mat44 (numpy.ndarray): A 4x4 transformation matrix used to transform the coordinates.

        Returns:
            list: A list [x, y] of the transformed point in the scan frame.
        """
        xyz = tuple(np.dot(mat44, np.array([coord[0],coord[1], 0, 1.0])))[:3]
        r = geometry_msgs.msg.PointStamped()
        r.point = geometry_msgs.msg.Point(*xyz)
        return [r.point.x,r.point.y]

    def cell_location_to_ind(self, cell_location):
        """
        Convert a cell location (x, y) in the grid to a linear index.

        Args:
            cell_location (list): A list [x, y] representing the cell location in the grid.

        Returns:
            int: The index (of the PDA) corresponding to the cell location.
        """
        return int((cell_location[1] * int(self.map_width)) + cell_location[0])

    def cell_location_to_x_y_M(self, cell_location):
        """
        Convert a cell location (x, y) in the grid to map coordinates (meters).

        Args:
            cell_location (tuple): A tuple (x, y) representing the cell location in the grid.

        Returns:
            list: A list [x, y] of the cell location in map coordinates (meters).
        """
        x = (cell_location[0] * self.map_resolution) + self.map_origin_x
        y = (cell_location[1] * self.map_resolution) + self.map_origin_y
        return [x, y]

    def ind_to_cell_location(self, index):
        """
        Convert a linear index to a cell location (x, y) in the grid.

        Args:
            index (int): The linear index of the cell.

        Returns:
            list: A list [x, y] representing the cell location in the grid.
        """
        cell_x = index % self.map_width
        cell_y = math.ceil((index - cell_x) / self.map_width)
        return [cell_x, cell_y]

    def x_y_M_to_cell_location(self, x_y_M):
        """
        Convert map coordinates (meters) to a cell location (x, y) in the grid.

        Args:
            x_y_M (list): A list [x, y] of the map coordinates (meters).

        Returns:
            list: A list [x, y] representing the cell location.
        """
        cell_x = math.ceil((x_y_M[0] - (self.map_origin_x)) / self.map_resolution)
        cell_y = math.ceil((x_y_M[1] - (self.map_origin_y)) / self.map_resolution)
        return [cell_x, cell_y]

    def mapmaker(self):
        """
        Creates the quadtree and zones for the map, and initializes the matrix (m_z) that 
        stores zone information.It clusters the free cells into zones using the KMeans algorithm,
        calculates zone centers, and inserts points into the quadtree for spatial organization.

        Args:
            None required. Utilizes instance attributes such as `self.free_cells`, `self.zones`, and others 
            which are assumed to be already set in the class instance.

        Returns:
            None. The method operates by side effects, updating class attributes like `self.cells_x_y`,
            `self.points_x_y_M`, `self.qtree`, `self.zone_info`, and `self.m_z` with the relevant information 
            for zone management and quadtree construction.
        """
        rospy.loginfo("Preparing Zones .....")
        for cell in self.free_cells:
            cell_x_y = self.ind_to_cell_location(cell)
            self.cells_x_y.append(cell_x_y)
            self.points_x_y_M.append(self.cell_location_to_x_y_M(cell_x_y))

        self.points = [Target(*cell) for cell in self.cells_x_y]
        for point in self.points:
            self.qtree.insert(point)

        fit_array = np.array(self.points_x_y_M)
        zones = KMeans(n_clusters=self.zones, n_init=5, max_iter=10, random_state=None).fit(fit_array)
        self.zone_info = np.zeros((len(zones.labels_), 2))
        self.zone_info[:, 0] = self.free_cells
        self.zone_info[:, 1] = zones.labels_
        src = 0
        while src < self.zones:
            carrier = np.where(zones.labels_ == src)
            self.m_z[src, 0] = zones.cluster_centers_[src][0]
            self.m_z[src, 1] = zones.cluster_centers_[src][1]
            self.m_z[src, 2] = len(carrier[0])
            self.m_z[src, 3] = 0
            self.m_z[src, 4] = 10e-20
            self.m_z[src, 5] = 1e20
            src += 1
        rospy.loginfo("Zones are prepared.")

    def tablemaker(self):
        """
        Creates a matrix (m_c) that updates cell coverage information to memory.
        The matrix maintains the information about cell indices, their coverage status, 
        and association with zones.

        Args:
            None required. Uses class attributes such as `self.rows`, `self.PDA`, `self.zone_info`, 
            and `self.m_c` that should be already initialized and populated.

        Returns:
            None. The method operates by updating the `self.m_c` matrix attribute with 
            the coverage information.

        Raises:
            AttributeError: If any required class attributes are not initialized before calling this method.
            IndexError: If array indexing goes beyond the limits of initialized arrays or matrices.
        """
        self.m_c[:, 0] = self.rows
        self.m_c[np.where(self.PDA == 100), 1] = -1
        self.m_c[np.where(self.PDA == -1), 1] = -1

        ind = np.isin(self.m_c[:, 0], self.zone_info[:, 0])
        self.m_c[ind, 1] = self.zone_info[:, 1]
        self.m_c[:, 2] = int(0)
        self.m_c[:, 3] = int(-1)

    def worker(self, TF_MS_t, query, S_F_origin, scan_angles, scan_ranges_t):
        """ Algorithm 5
        Updates the cell coverage based on sensor data.

        Args:
            TF_MS_t (numpy.ndarray): The transformation matrix from map to scan frame.
            query (list of tuple): A list of cell coordinates (x, y) to query for coverage.
            S_F_origin (tuple): The origin point (x, y) in the scan frame.
            scan_angles (list): A list of scan angles corresponding to sensor data.
            scan_ranges_t (list): A list of scan ranges corresponding to sensor data.

        Returns:
            None. This method updates the `self.m_c` and `self.m_z` matrices by marking cells as covered
            and calculating the updated coverage percentage for the zones.

        Raises:
            IndexError: If an index lookup goes beyond the matrix dimensions, typically when the 
                        transformation results in a cell index that doesn't exist in the matrix.
            ValueError: If a computed angle does not exist in the `scan_angles` list, which would imply
                        that there's a mismatch between the sensor data and the computed angles.
        """
        try:
            for found_point in query:
                index = self.cell_location_to_ind(found_point)
                if self.m_c[index, 2]==0:
                    X_Y_M = self.cell_location_to_x_y_M(found_point)
                    X_Y_S = self.transform(X_Y_M , TF_MS_t)
                    alpha = (int(math.degrees(math.atan2(X_Y_S[1], X_Y_S[0]))) + 360) % 360

                    try:
                        if scan_angles.index(alpha) >= 0:                       #determine if scan angle matches to cell orientation
                            dist = self.get_distance(X_Y_S, S_F_origin)
                            if dist <= scan_ranges_t[scan_angles.index(alpha)]: #determine if cell distance matches scan range
                                self.m_c[index, 2] = 1
                                src = int(self.m_c[index, 1])
                                self.m_z[src, 3] = self.m_z[src, 3] + 1
                                self.m_z[src, 4] = (float(self.m_z[src, 3]) / self.m_z[src, 2]) * 100

                    except ValueError:
                        continue

        except IndexError:
            pass

    def Coverage_Calculator(self,TF_MS_t, x0, y0, z0, grid_range, scan_angles,scan_ranges_t):
        """
        Calculates and updates the coverage map based on the robot's position and scan data. It determines
        which cells are within the robot's scanning range and updates their coverage status using multithreading
        if enabled.

        Args:
            TF_MS_t (numpy.ndarray): The transformation matrix from map to scan frame at time t.
            x0 (float): The x-coordinate of the robot's position in the map frame.
            y0 (float): The y-coordinate of the robot's position in the map frame.
            z0 (float): The z-coordinate of the robot's position in the map frame (unused).
            grid_range (int): The range of the grid cells to be queried around the robot's position.
            scan_angles (list of float): The angles at which scan data is available.
            scan_ranges_t (list of float): The range measurements corresponding to `scan_angles`.

        Returns:
            None. This function updates internal state matrices related to coverage, and publishes data
            related to the current coverage rate and next target zone.

        Raises:
            Exception: If multithreading is enabled but threads fail to execute properly.
            rospy.ROSException: If the transformation listeners timeout while waiting for transformations.
        """
        Cell_X_Y = self.x_y_M_to_cell_location([x0, y0])
        center, radius = [Cell_X_Y[0], Cell_X_Y[1]], grid_range
        query = []
        center = Target(*center)
        query = np.array(self.qtree.query_radius(center, radius, query))
        S_F_origin = [0,0] #scan frame origin

        if self.Multi_threading:
            scope = []
            FOV = int(len(query) / float(self.NOT))
            for i in range(0,self.NOT):
                if i==self.NOT-1:
                   scope.append(threading.Thread(target=self.worker, args=(TF_MS_t,query[FOV*i:], S_F_origin, scan_angles, scan_ranges_t)))
                   scope[-1].start()

                scope.append(threading.Thread(target=self.worker, args=(TF_MS_t,query[FOV*i:FOV*(i+1)], S_F_origin, scan_angles, scan_ranges_t)))
                scope[-1].start()

        else:
            self.worker(TF_MS_t, query, S_F_origin, scan_angles, scan_ranges_t)

        set_time = rospy.Time(0)
        self.listener.waitForTransform('map', 'base_footprint', set_time, rospy.Duration(600))
        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', set_time)
        x = trans[0]
        y = trans[1]
        src = 0
        count = 1e20
        mini = min(self.m_z[:, 4])
        mlist = np.where( (self.m_z[:, 4]<=mini) )

        for i in mlist[0]:
            self.m_z[i, 5] = self.get_distance([self.m_z[i, 0], self.m_z[i, 1]], [x, y])
            if self.m_z[i, 5] < count:
                count = self.m_z[i, 5]
                src = i

        self.x_y_zones.data = [self.m_z[src, 0], self.m_z[src, 1]]
        self.pub.publish(self.x_y_zones)
        current_cov_rate = (np.sum(self.m_z[:, 3]) / float(len(self.free_cells))) * 100
        self.pub4.publish(current_cov_rate)


        if round(current_cov_rate) >= self.dc:
            rospy.loginfo("Desired coverage is reached. Chaotic coverage path planner stopping.....")
            self.pub2.publish("stop")
            self.pub3.publish("finished")

    def shutdown(self, msg):
        """
        Subscriber callback. Used to signal the shutdown of the logistician process. Updates the internal state to indicate
        that the logistician should finish its loop.

        Args:
            msg (std_msgs.String): A message containing data that signals the shutdown process.
        """
        self.finish = msg.data

    def start(self, msg):
        """
        Publishes the string message to start and stop the chaotic trajectory planner.

        Args:
            msg (std_msgs.String): A message containing "start" or "stop" to control the robot's motion.
        """
        self.pub2.publish(msg)

    def logistician(self):
        """Algorithm 4
        Continuously updates the coverage map as the robot moves. It calls the `Coverage_Calculator`
        function in a loop, effectively tracking the coverage process in real-time.

        Args:
            None required. Utilizes instance attributes such as `self.scan` and `self.map_resolution`
            which should be set prior to calling this function.

        Raises:
            rospy.ROSException: If transformation listeners time out or if the coverage calculation
                                encounters any errors.
        """
        scan_angles = self.rad_to_deg(self.scan)
        grid_range = int(round(rospy.get_param("sensing_range") / self.map_resolution))
        #grid_range = int(round(self.scan.range_max / self.map_resolution))
        scan_frame = rospy.get_param("scan_frame")
        self.Multi_threading = rospy.get_param("Multi-threading")

        while self.finish != "finished":
            set_time = rospy.Time(0)
            self.listener.waitForTransform(scan_frame, 'map', set_time, rospy.Duration(600))
            (trans, rot) = self.listener.lookupTransform(scan_frame, 'map', set_time)
            TF_MS_t = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            scan = self.scan
            scan_ranges_t = scan.ranges
            self.listener.waitForTransform('map', scan_frame, set_time, rospy.Duration(600))
            (X_Y_S_M_t, rot) = self.listener.lookupTransform('map', scan_frame, set_time) #Get pose of sensor at time t

            x_S_M = X_Y_S_M_t[0]
            y_S_M = X_Y_S_M_t[1]
            z_S_M = 0.0

            self.Coverage_Calculator(TF_MS_t, x_S_M, y_S_M, z_S_M, grid_range, scan_angles, scan_ranges_t)


if __name__ == "__main__":
    try:
        map = map_zoner()
        rospy.Subscriber("/scan", LaserScan, map.get_scan)

        """subscribed to topic responsible for stopping Logistician"""
        rospy.Subscriber("/shutdown", String, map.shutdown)

        map.mapmaker()
        map.tablemaker()

        """Uses publisher to start chaotic motion (i.e ATP function)"""
        map.start("start")
        map.logistician()


    except rospy.ROSInterruptException:
        pass
