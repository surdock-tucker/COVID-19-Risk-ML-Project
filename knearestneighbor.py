import numpy as np
import scipy.spatial


# class for creating a k-nearest neighbor model
class KNearestNeighbor(object):

    def __init__(self, k, x, y):
        self.k = k
        self.y_train = y
        self.x_train = x

    def predict(self, x):

        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        np.zeros((num_test, num_train))
        distances = scipy.spatial.distance_matrix(x, self.x_train)

        num_test = distances.shape[0]
        y_prediction = np.zeros(num_test)
        for i in range(num_test):
            k_nearest_ids = np.argsort(distances[i, :])[:self.k]
            closest_y = self.y_train[k_nearest_ids]
            y_prediction[i] = np.argmax(np.bincount(closest_y))
        return y_prediction

    def get_k_value(self):
        return self.k
