"""
File: QUADTREE.PY
Author: Uyiosa Philip Amadasun

Brief:
        Slightly editted the original script to be interpreted and compiled in Python2.7.
    This module implements a quadtree data structure for efficient spatial querying in two dimensions. 
    The quadtree enables quick access to map coordinates, which is neccessary for coverage calculation 
    and obstacle avoidance.
    
    Classes included:
    - Target: Represents a point in 2D space, can carry a payload.
    - Rect: Represents a rectangular region in 2D space.
    - QuadTree: Implements the quadtree with methods for inserting points 
      and querying regions.

References:
    [1] CHRISTIAN, Quadtree implementation in python, 2020. Accessed 2023.
        https://scipython.com/blog/quadtrees-2-implementation-in-python.
     
"""
import numpy as np

""" variable c is not relevant to the class methods we actually use for our application."""
c=0

class Target:
    """
    Represents a point in a 2D space.
    
    Attributes:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        payload (Optional): Additional data associated with the point.
    """

    def __init__(self, x, y, payload=None):
        """
        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.
            payload (Optional): Additional data to associate with the point.
        """
        self.x, self.y = float(x), float(y)
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))
    def __str__(self):
        return 'P({:.2f}, {:.2f})'.format(self.x, self.y)

    def distance_to(self, other):
        """
        Calculate the Euclidean distance between this point and another point.
        
        Args:
            other (Target or tuple): Another point or a tuple of coordinates.
            
        Returns:
            float: The Euclidean distance between two points.
        """
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)

class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        """
        Args:
            cx, cy (float): Coordinates of the center of the rectangle.
            w, h (float): Width and height of the rectangle.
        """
        self.cx, self.cy = float(cx), float(cy)
        self.w, self.h = float(w), float(h)
        self.west_edge, self.east_edge = (cx - (float(w)/2)), (cx + (float(w)/2))
        self.north_edge, self.south_edge = (cy - (float(h)/2)), (cy + (float(h)/2))

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                    self.north_edge, self.east_edge, self.south_edge)

    def contains(self, point):
        """
        Check if a point is inside this rectangle.
        
        Args:
            point (Target or tuple): A point or a tuple of coordinates.
            
        Returns:
            bool: True if the point is inside the rectangle, False otherwise.
        """

        try:
            point_x, point_y = (point.x), (point.y)
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x <= self.east_edge and
                point_y >= self.north_edge and
                point_y <= self.south_edge)

    def intersects(self, other):
        """
        Check if another rectangle intersects with this rectangle.
        
        Args:
            other (Rect): Another rectangle.
            
        Returns:
            bool: True if rectangles intersect, False otherwise.
        """
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
                sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
                sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = (self.boundary.cx), (self.boundary.cy)
        w, h = (float(self.boundary.w) / 2), (float(self.boundary.h) / 2)
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(Rect(cx - (float(w)/2), cy - (float(h)/2), w, h),
                                    self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + (float(w)/2), cy - (float(h)/2), w, h),
                                    self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + (float(w)/2), cy + (float(h)/2), w, h),
                                    self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - (float(w)/2), cy + (float(h)/2), w, h),
                                    self.max_points, self.depth + 1)
        self.divided = True

    def insert(self, point):
        global c
        """
        Insert a point into the quadtree.
        
        Args:
            point (Target): The point to insert.
            
        Returns:
            bool: True if the point is inserted successfully, False otherwise.
        """

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            #c+=1
            #print("(",point.x,",",point.y," )")
            #print(c)
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """
        Find points in the quadtree that lie within a boundary.
        
        Args:
            boundary (Rect): The boundary to search points within.
            found_points (list): List to store found points.
            
        Returns:
            list: List of points found within the boundary.
        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            #print("NO INTERSECT")
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points


    def query_circle(self, boundary, centre, radius, found_points):
        """
        Find points in the quadtree within a radius of a centre point.
        
        Args:
            boundary (Rect): Boundary that contains the search circle.
            centre (Target): Centre point of the search circle.
            radius (float): Radius of the search circle.
            found_points (list): List to store found points.
            
        Returns:
            list: List of points found within the search circle.
        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            #print("NO INTERSECT")
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append([point.x,point.y]) #point

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """
        Find points in the quadtree that lie within a radius of a centre point.
        
        Args:
            centre (Target): Centre point of the search circle.
            radius (float): Radius of the search circle.
            found_points (list): List to store found points.
            
        Returns:
            list: List of points found within the radius of the centre point.
        """
        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(centre.x,centre.y, 2*radius, 2*radius)
        return self.query_circle(boundary, centre, radius, found_points)

    """
    These last two methods are irrelevant to our application
    """
    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)
