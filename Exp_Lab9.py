import numpy as np

# Step 1: Function to perform PCA
def pca(data, num_components=1):
    # Mean centering the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Step 2: Calculate the covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort the eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Step 5: Select the top eigenvectors (principal components)
    top_eigenvectors = sorted_eigenvectors[:, :num_components]
    
    # Step 6: Ensure the eigenvector has a consistent sign
    top_eigenvectors *= np.sign(top_eigenvectors[0, :])
    
    # Step 7: Project the data onto the top eigenvectors
    reduced_data = centered_data.dot(top_eigenvectors)
    
    return reduced_data, top_eigenvectors

# Step 2: Function to get input points from the user
def get_input_points():
    points = []
    print("Enter points for two features (Feature1, Feature2). Type 'done' to stop.")
    while True:
        point = input("Enter point (Feature1, Feature2): ")
        if point.lower() == 'done':
            break
        try:
            feature1, feature2 = map(float, point.split(','))
            points.append([feature1, feature2])
        except ValueError:
            print("Invalid input. Please enter two floats separated by a comma.")
    return np.array(points)

def get_input_points_programmatically(points_list):
    points = []
    for point in points_list:
        try:
            feature1, feature2 = map(float, point.split(','))
            points.append([feature1, feature2])
        except ValueError:
            print("Invalid input. Please enter two floats separated by a comma.")
    return np.array(points)

# Step 3: Main function to run the PCA
def main():
    # Get input points
    data = get_input_points()
    
    # Check if the input data has at least two features
    if data.shape[1] < 2:
        print("Error: Input data must have at least two features.")
        return
    
    # Perform PCA to reduce from 2D to 1D
    reduced_data, principal_components = pca(data, num_components=1)
    
    # Display the results
    print("\nReduced Data (1D):")
    print(reduced_data)
    
    print("\nPrincipal Component (Eigenvector for 1D):")
    print(principal_components)
    
    print("\nMean of the original data:")
    print(np.mean(data, axis=0))

# Step 4: Run the main function
if __name__ == "__main__":
    main()

