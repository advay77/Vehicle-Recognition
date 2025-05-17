import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    """
    A simple centroid-based tracker for tracking objects across frames
    """
    def __init__(self, max_disappeared=50, max_distance=50):
        # Initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object ID
        # to its centroid and number of consecutive frames it has been
        # marked as "disappeared"
        self.next_object_id = 0
        self.objects = OrderedDict()  # {ID: (box, class_id, centroid, confidence)}
        self.disappeared = OrderedDict()  # {ID: count}
        self.previous_centroids = {}  # {ID: centroid}
        
        # Store the number of maximum consecutive frames an object is
        # allowed to be marked as "disappeared" until we deregister it
        self.max_disappeared = max_disappeared
        
        # Store the maximum distance between centroids to associate an
        # object -- if the distance is larger than this maximum distance
        # we'll start to mark the object as "disappeared"
        self.max_distance = max_distance
    
    def _calculate_centroid(self, box):
        """Calculate centroid from bounding box"""
        x, y, w, h = box
        return (int(x + w/2), int(y + h/2))
    
    def register(self, box, class_id, confidence):
        """Register a new object"""
        centroid = self._calculate_centroid(box)
        self.objects[self.next_object_id] = (box, class_id, centroid, confidence)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.previous_centroids:
            del self.previous_centroids[object_id]
    
    def update(self, detections):
        """
        Update the tracker with new detections
        
        Args:
            detections: List of tuples (box, class_id, confidence)
            
        Returns:
            OrderedDict of tracked objects {ID: (box, class_id, centroid, confidence)}
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        # Initialize an array of input centroids for the current frame
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        input_boxes = []
        input_class_ids = []
        input_confidences = []
        
        # Process detections
        for i, (box, class_id, confidence) in enumerate(detections):
            centroid = self._calculate_centroid(box)
            input_centroids[i] = centroid
            input_boxes.append(box)
            input_class_ids.append(class_id)
            input_confidences.append(confidence)
        
        # If we are currently not tracking any objects, register all
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(input_boxes[i], input_class_ids[i], input_confidences[i])
        
        # Otherwise, try to match input centroids to existing object centroids
        else:
            # Get IDs and centroids of current objects
            object_ids = list(self.objects.keys())
            object_centroids = [obj[2] for obj in self.objects.values()]
            
            # Save current centroids for speed calculation
            for object_id, obj in self.objects.items():
                self.previous_centroids[object_id] = obj[2]
            
            # Compute distances between each pair of object centroids and input centroids
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Find the smallest value in each row and sort the row indexes based on their
            # minimum values so that we can grab the row indexes with the smallest distances first
            rows = D.min(axis=1).argsort()
            
            # Find the smallest value in each column and sort using the previously
            # computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # Keep track of which object IDs and inputs we've already examined
            used_rows = set()
            used_cols = set()
            
            # Loop over the combinations of (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # If we've already examined this row or column, skip it
                if row in used_rows or col in used_cols:
                    continue
                
                # If the distance is greater than the maximum distance, don't associate
                if D[row, col] > self.max_distance:
                    continue
                
                # Otherwise, grab the object ID for the current row, reset its
                # disappeared counter, and update its centroid
                object_id = object_ids[row]
                box = input_boxes[col]
                class_id = input_class_ids[col]
                centroid = input_centroids[col]
                confidence = input_confidences[col]
                
                self.objects[object_id] = (box, class_id, centroid, confidence)
                self.disappeared[object_id] = 0
                
                # Indicate we've examined this row and column
                used_rows.add(row)
                used_cols.add(col)
            
            # Compute the row and column indices we haven't yet examined
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)
            
            # If we have more objects than inputs, check if any objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    # Grab the object ID and increment the disappeared counter
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Check if we should deregister the object
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Otherwise, register each new input centroid as a trackable object
            else:
                for col in unused_cols:
                    self.register(input_boxes[col], input_class_ids[col], input_confidences[col])
        
        # Return the set of trackable objects
        return self.objects