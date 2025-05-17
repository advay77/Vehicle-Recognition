import cv2

class Settings:
    """
    Configuration settings for the vehicle detection system
    """
    def __init__(self):
        # Detection settings
        self.CONFIDENCE_THRESHOLD = 0.5
        self.NMS_THRESHOLD = 0.4
        
        # Vehicle classes to detect
        self.VEHICLE_CLASSES = ['car', 'motorbike', 'bus', 'truck']
        
        # Colors for different vehicle classes (BGR format)
        self.COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
        
        # Font for text display
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        
        # Position of counting line (fraction of frame height)
        self.LINE_Y_POSITION = 0.6
        
        # Speed estimation settings
        self.PIXELS_PER_METER = 8  # Calibration factor (approximate)
        self.SPEED_THRESHOLD = 30  # km/h threshold for fast/slow classification
        
        # Tracking settings
        self.MAX_DISAPPEARED = 40  # Maximum number of frames to keep tracking a disappeared object
        self.MAX_DISTANCE = 50     # Maximum distance between centroids to associate objects