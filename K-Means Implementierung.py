"""
Docstring: Aufgabe 2: k-Means
This program contains a implementation of the k-means clustering algorithm in Python.

The k-means algorithm is a method for finding clusters in a dataset by iteratively
updating the centroids of the clusters until they are stabilized. The algorithm starts
by selecting k random points from the dataset as the initial centroids of the clusters.
For each point in the dataset, it then calculates the distance to each centroid and
assigns the point to the cluster whose centroid is nearest. The centroids of the clusters
are then updated by calculating the mean position of all the points in the cluster.
The process is repeated until the centroids do not change position anymore or until a
maximum number of iterations is reached.


The program contains the following functions:
distance: Calculates the distance between two points or between a point and a list of points.
k_means: Implements the k-means clustering algorithm to find the centroids of n_clusters clusters
in a dataset.
evaluate: Classifies each point in a dataset into a cluster based on the nearest centroid.
testing: Loads the data from the data.csv file and prints the points and labels.
visualize: Plots the points and centroids using matplotlib.
main: The main function that calls the other functions to execute the k-means algorithm.
"""


import numpy as np
import matplotlib.pyplot as plt


# Aufgabe 1.
def distance(point, data):
    """
    Docstring:
    This function calculates the distance between two points or between a point
    and a list of points.
    
    Parameters:
    point: A single point represented as a 1-dimensional numpy array.
    data: A list of points represented as a 2-dimensional numpy array.
    
    Returns:
    float or numpy array: If data is a single point, returns the distance between
    point and data. If data is a list of points, returns an array of distances
    between point and each point in the list.
    """
    
    # In case data is a single point, meaning if the shape is 1-dimensional,
    # the distance between point and data will be calculated.
    # The shape attribute returns a tuple that represents the dimensions.
    if len(data.shape) == 1:
        # Using numpy's sum and sqrt functions to calculate the sum of the
        # squared differences between the x and y coordinates of point and data
        # (x1 - x2)^2 + (y1 - y2)^2, and then taking the square root of the
        # result to get the Euclidean distance.
        return np.sqrt(np.sum((point - data) ** 2))
    # In case data is a list of points, meaning the shape is 2-dimensional,
    # the distance between point and each point in the list will be calculated.
    else:
        distances_list = []
        # Using a for loop to iterate over each point in the list data.
        # For each point p, it calculates the distance between point and p
        # using the same formula as above and appends the result to the distances list.
        for p in data:
            distances_list.append(np.sqrt(np.sum((point - p) ** 2)))
        # The np.array function converts a list of values into a numpy array.
        return np.array(distances_list)




# Aufgabe 2.
def k_means(data, n_clusters, max_iter):
    """
    Docstring:
    This function implements the k-means clustering algorithm to find the centroids
    of n_clusters clusters in a dataset.
    
    Parameters:
    data: A 2-dimensional numpy array representing the dataset. Each row is a point in the dataset.
    n_clusters: An integer representing the number of clusters to be formed.
    max_iter: An integer representing the maximum number of iterations to be performed by the algorithm.
    
    Returns:
    numpy array: A 2-dimensional numpy array containing the centroids of the clusters.
    """
    
    # This line of code initializes the centroids for the clusters by randomly selecting n_clusters
    # points from the data array and storing them in the centroids variable.
    # The data.shape[0] attribute returns the number of rows in the data array,
    # and the np.random.choice function is used to randomly select n_clusters rows from this range
    # without replacement meaning that the same row will not be selected more than once.
    centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
    # Keeping track of the number of iterations that will be performed. 
    n_iter = 0
    # Keeping track of whether the centroids will have changed position during the current iteration.
    changed = True

    
    # While the centroids have changed and the number of iterations is less than max_iter.
    while changed and n_iter < max_iter:
        # Initializing a list to store the indices of the nearest centroids for each point.
        nearest_centroids = []

        for point in data:
            # Calculating the distances between the point and each centroid.
            distances = distance(point, centroids)
            # The np.argmin function returns the index of the minimum value in an array.
            # In this case, it is used to find the index of the centroid that is nearest to the point
            # by returning the index of the minimum distance in the distances array.
            # This index is then appended to the nearest_centroids list.
            nearest_centroids.append(np.argmin(distances))
            
        # Converting the nearest_centroids list into a numpy array.
        nearest_centroids = np.array(nearest_centroids)
        # Storing the new centroids.
        new_centroids = []

        
        for c in range(n_clusters):
            # Using to update the position of the centroid for the current cluster.
            cluster_points = data[nearest_centroids == c]
            
            # If the cluster is not empty, calculating the mean of the points to get the new centroid.
            if cluster_points.shape[0] > 0:
                # The axis=0 parameter specifies that the mean should be calculated along the
                # rows (i.e., across the columns).
                # The mean position is then appended to the new_centroids list using the append method.
                new_centroids.append(np.mean(cluster_points, axis=0))
                
            # Else if the cluster is empty, we use the previous centroid as the new centroid.
            # This is done to prevent the algorithm from getting stuck in an infinite loop
            # if a cluster becomes empty during the iterations.
            else:
                new_centroids.append(centroids[c])
                
        # Converting the new_centroids list into a numpy array.
        new_centroids = np.array(new_centroids)
        
        # If the new centroids are different from the previous centroids, setting the changed flag to True.
        if not np.array_equal(centroids, new_centroids):
            changed = True
            # Updating the centroids with the new centroids.
            centroids = new_centroids
            
        # If the new centroids are the same as the previous centroids, setting the changed flag to False.
        else:
            changed = False
            
        # Using to keep track of the number of iterations that have been performed by the k_means function.
        n_iter += 1
        
    return centroids




# Aufgabe 3.
def evaluate(centroids, points):
    """
    Docstring:
    This function evaluates the closest centroid for each point in a list of points.
    
    Parameters:
    centroids: A list of centroids.
    points: A list of points to be evaluated.
    
    Returns:
    list: A list of tuples containing the centroid and index of the closest centroid
    for each point.
    """
    
    # Soring the results.
    results = []
    
    for point in points:
        # Initializing variables to store the distance and index of the closest centroid.
        min_distance = float('inf')
        min_index = -1

        # Using a for loop to iterate over each centroid in the list centroids.
        for i, centroid in enumerate(centroids):
            # Calculating the distance between point and centroid.
            distance = np.sqrt(np.sum((point - centroid) ** 2))
            
            # If the distance is smaller than the current minimum distance,
            # update the minimum distance and index.
            if distance < min_distance:
                min_distance = distance
                min_index = i
                
        # Appending the centroid and index for the point to the results list.
        results.append((centroids[min_index], min_index))

    return results




# Aufgabe 4.
def testing():
    """
    Docstring:
    This function loads a dataset from a CSV file and prints the x and y coordinates and
    labels for each point in the dataset.
    """
    
    print("\n\n\n\nAufgabe 4.: Die ersten beiden Spalten enthalten die x- und y-Koordinaten")
    print("der Punkte, die dritte Spalte das Label.")

    data = np.loadtxt('data.csv', delimiter=',')

    points = data[:, :2]  # The first two columns contain the x and y coordinates.
    labels = data[:, 2]   # The third column contains the labels.


    # Looping through the rows of the points and labels arrays.
    for point, label in zip(points, labels):
        x, y = point  # Unpacking the point coordinates.
        output_string = 'x: {:.2f}, y: {:.2f}, label: {}'  # Specify the format string.
        output = output_string.format(x, y, label)  # Using the format method to format the output.
        print(output)  




# Aufgabe 5.
def visualize(points, centroids, labels):
    """
    Docstring:
    This function visualizes the points and centroids of a dataset using matplotlib.
    
    Parameters:
    points: A list of points in the dataset.
    centroids: A list of centroids.
    labels: A list of labels for the points.
    
    Returns:
    None: The function displays a scatter plot of the points using the x- and y-coordinates
    from the points list and the labels as the color values, and a plot of the centroids
    using the x- and y-coordinates from the centroids list and the 'k+' marker style.
    """
    
    # Using the scatter function from matplotlib to plot the points, and
    # using the x- and y-coordinates from the points array and the labels as the color values.
    plt.scatter(x=[point[0] for point in points], y=[point[1] for point in points], c=labels)
    # Using the x- and y-coordinates from the centroids list and the 'k+' marker style.
    plt.plot([centroid[0] for centroid in centroids], [centroid[1] for centroid in centroids], 'k+', markersize=10)
    # Showing the plot.
    plt.show()




def main():
    """
    Docstring:
    This main function of the program performs the following tasks:
    It Loads the data from the 'data.csv' file using the np.loadtxt function and stores it in a numpy array.
    Extracts the points from the data array.
    Tests the distance function with different input parameters.
    Tests the k_means function with different input parameters.
    Tests the evaluate function with different input parameters.
    Calls the testing function.
    Calls the visualize function with the points, centroids, and labels as input.
    """
    
    # Aufgabe 1.
    # Using np.loadtxt to load the data from the CSV file and to store it in numpy array.
    # The second parameter is the delimiter that separates the data values in the file,
    # in this case, a comma.
    data = np.loadtxt('data.csv', delimiter=',')
    # Extracting the points from the data array that selects the first two columns of the array.
    # This is useful in this case because the data points in the CSV file contain three columns:
    # two for the x and y coordinates of the points, and one for the label (class) of the points.
    points = data[:,:2]

    # Testcases
    print("Testfälle für die Aufgabe 1.")
    # Calculating the distance between the first and second points.
    print("Punkt 1:", points[0], "Punkt 2:", points[1])
    print("Distanz zwischen dem ersten und zweiten Punkt:", distance(points[0], points[1]))
    # Calculating the distance between the first point and a list of points.
    print("\nDistanz zwischen dem ersten Punkt und einer Liste von Punkten:\n", distance(points[0], points[1:]))
    # Calculating the distance between two lists of points.
    print("\nDistanz zwischen zwei Listen von Punkten:\n", distance(points[:5], points[5:]))


    # Aufgabe 2.
    # Testcases
    print("\n\n\n\nTestfälle für die Aufgabe 2.")
    # Setting the number of clusters to 3.
    n_clusters = 3
    # Setting the maximum number of iterations to 11.
    max_iter = 11
    # Calling the k_means function to cluster the points.
    centroids = k_means(points, n_clusters, max_iter)
    print("Die ermittelten Centroide der Cluster lauten:\n", centroids)
    n_clusters = 5
    max_iter = 5
    centroids = k_means(points, n_clusters, max_iter)
    print("\nDie ermittelten Centroide der Cluster lauten:\n", centroids)
    n_clusters = 4
    max_iter = 20
    centroids = k_means(points, n_clusters, max_iter)
    print("\nDie ermittelten Centroide der Cluster lauten:\n", centroids)


    # Aufgabe 3.
    # Testcases
    print("\n\n\n\nTestfälle für die Aufgabe 3.")
    # Clustering the points into 2 clusters using the k_means function.
    centroids = k_means(points, 2, 10)
    # Classifing the points using the evaluate function.
    classified_points = evaluate(centroids, points)
    for point, (centroid, index) in zip(points, classified_points):
        print(f"Punkt: {point} Centroid: {centroid} Index: {index}")


    # Aufgabe 4.
    testing()

    
    # Aufgabe 5.
    data = np.loadtxt('data.csv', delimiter=',')
    points = data[:,:2]
    labels = data[:,2]
    centroids = k_means(points, 2, 10)
    classified_points = evaluate(centroids, points)
    classified_labels = [label for _, label in classified_points]
    # Visualizing the points and centroids.
    visualize(points, centroids, classified_labels)
    



if __name__ == "__main__":
    main()


















