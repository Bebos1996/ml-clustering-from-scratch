import numpy as np
class centroid:
    def __init__(self, dataset, columns, class_name):
        self.coordinates = np.array([])
        for column in columns:
            if column != class_name:
                self.coordinates = np.append(self.coordinates, np.random.uniform(low=dataset[column].min(), high=dataset[column].max()))

    def distance(self, instance):
        dist = 0
        for i in range(len(self.coordinates)):
            dist += np.sqrt(np.power(self.coordinates[i] - instance[i],2))
        return dist