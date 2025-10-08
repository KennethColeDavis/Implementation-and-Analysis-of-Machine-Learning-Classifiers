from math import sqrt
class simpleKNN:

    @staticmethod
    def calculate_distance(point1, point2):
        point1 = point1[0]
        point2 = point2[0]
        distance = 0

        for i in range (len(point1)):
            diff = (point1[i] -point2[i]) ** 2
            distance = distance + diff
        
        return sqrt(distance) 
    
    @staticmethod
    def get_neighbors(trainData, testPoint, number_neighbors):
        distances = [
            (trainingPoint, simpleKNN.calculate_distance(testPoint,trainingPoint)) 
            for trainingPoint in trainData
        ]

        distances.sort(key=lambda tup: tup[1])

        neighbors = [distances[i][0] for i in range(number_neighbors)]

        return neighbors
    
    @staticmethod
    def predict_classification(trainData, testsPoint, number_neighbors):

        neighbors = simpleKNN.get_neighbors(trainData, testsPoint, number_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key = output_values.count) 

        return prediction
    




    
