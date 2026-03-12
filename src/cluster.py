import numpy as np
class cluster:
    def __init__(self, cr):
        self.centroid = self.old_centroid = cr
        self.assigned = self.silhouettes = []
        self.changed = True
        self.label = ""

    def new_centroid(self):
        self.old_centroid = self.centroid
        num_features = len(self.assigned[0])-1
        new_centroid = [0 for _ in range(num_features)]
        num_instances = len(self.assigned)
        for i in range(0, num_instances):
            for j in range(0, num_features):
                new_centroid[j] += self.assigned[i][j]/num_instances
        if self.centroid.distance(new_centroid) != 0:
            self.changed = True
            self.centroid.coordinates = np.array(new_centroid)
        else:
            self.changed = False
    
    def __str__(self) -> str:
        return self.assigned.__str__()