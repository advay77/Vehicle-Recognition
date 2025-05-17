import math
from collections import deque

class SpeedEstimator:
    """
    Class to estimate the speed of tracked objects
    """
    def __init__(self, fps=30, ppm=8, max_points=10):
        """
        Initialize the speed estimator
        
        Args:
            fps: Frames per second of the video
            ppm: Pixels per meter (calibration factor)
            max_points: Maximum number of points to keep in history
        """
        self.fps = fps
        self.ppm = ppm  # pixels per meter
        self.max_points = max_points
        self.positions = {}  # {object_id: deque of positions}
        self.speeds = {}  # {object_id: current speed in km/h}
    
    def update(self, object_id, position):
        """
        Update the position history of an object
        
        Args:
            object_id: ID of the tracked object
            position: (x, y) tuple of the object's position
        """
        if object_id not in self.positions:
            self.positions[object_id] = deque(maxlen=self.max_points)
        
        self.positions[object_id].append(position)
        
        # Calculate speed if we have enough positions
        if len(self.positions[object_id]) >= 2:
            self.calculate_speed(object_id)
    
    def calculate_speed(self, object_id):
        """
        Calculate the speed of an object based on its position history
        
        Args:
            object_id: ID of the tracked object
        """
        positions = self.positions[object_id]
        
        if len(positions) < 2:
            self.speeds[object_id] = 0
            return
        
        # Calculate distance in pixels
        pixel_distance = 0
        for i in range(1, len(positions)):
            pixel_distance += self._calculate_distance(positions[i-1], positions[i])
        
        # Convert to meters
        meters = pixel_distance / self.ppm
        
        # Calculate time elapsed in seconds
        time_elapsed = (len(positions) - 1) / self.fps
        
        # Calculate speed in m/s and convert to km/h
        if time_elapsed > 0:
            speed_ms = meters / time_elapsed
            speed_kmh = speed_ms * 3.6
            
            # Apply smoothing (simple moving average)
            if object_id in self.speeds:
                # 70% new speed, 30% old speed for smoothing
                self.speeds[object_id] = 0.7 * speed_kmh + 0.3 * self.speeds[object_id]
            else:
                self.speeds[object_id] = speed_kmh
        else:
            self.speeds[object_id] = 0
    
    def get_speed(self, object_id):
        """
        Get the current speed of an object
        
        Args:
            object_id: ID of the tracked object
            
        Returns:
            Speed in km/h
        """
        return self.speeds.get(object_id, 0)
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def reset(self, object_id=None):
        """
        Reset the speed estimator for a specific object or all objects
        
        Args:
            object_id: ID of the tracked object, or None to reset all
        """
        if object_id is None:
            self.positions = {}
            self.speeds = {}
        elif object_id in self.positions:
            del self.positions[object_id]
            if object_id in self.speeds:
                del self.speeds[object_id]