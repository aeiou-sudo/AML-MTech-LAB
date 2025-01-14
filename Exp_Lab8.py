import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# DBSCAN Algorithm
def dbscan(points, epsilon, min_points):
    labels = [-1] * len(points)  # -1 denotes noise
    cluster_id = 0
    border_points = set()

    def region_query(point_idx):
        neighbors = []
        for i, p in enumerate(points):
            if euclidean_distance(points[point_idx], p) <= epsilon:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # Mark it as part of the cluster
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_points:
                    neighbors.extend(new_neighbors)
            i += 1

    for i in range(len(points)):
        if labels[i] != -1:  # Skip if already processed
            continue

        neighbors = region_query(i)
        if len(neighbors) >= min_points:
            expand_cluster(i, neighbors, cluster_id)
            cluster_id += 1
        else:
            labels[i] = -1  # Mark as noise

    # Identifying border points
    for i in range(len(points)):
        if labels[i] != -1:  # If not noise
            neighbors = region_query(i)
            if len(neighbors) >= 1 and len(neighbors) < min_points:
                border_points.add(i)
    
    return labels, border_points

# Get points from the user
def get_points():
    points = []
    print("Enter points as (x, y) coordinates. Type 'done' to stop.")
    while True:
        point = input("Enter point (x, y): ")
        if point.lower() == "done":
            break
        try:
            x, y = map(int, point.split(","))
            points.append((x, y))
        except ValueError:
            print("Invalid input, please enter in the format x,y.")
    return points

# Main function
def main():
    points = get_points()
    epsilon = float(input("Enter epsilon (ε): "))
    min_points = int(input("Enter minimum points (min_points): "))
    
    # Perform DBSCAN clustering
    labels, border_points = dbscan(points, epsilon, min_points)

    # Display results
    clusters = {}
    for idx, label in enumerate(labels):
        if label != -1:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(points[idx])

    # Show clusters and noise points
    print("\nClusters:")
    for cluster_id, cluster_points in clusters.items():
        print(f"Cluster {cluster_id}: {cluster_points}")
    
    noise_points = [points[i] for i in range(len(points)) if labels[i] == -1]
    if noise_points:
        print(f"\nNoise points: {noise_points}")
    else:
        print("\nNo noise points detected.")

    # Show border points
    print("\nBorder points:")
    border_points_list = [points[i] for i in border_points]
    if border_points_list:
        print(border_points_list)
    else:
        print("No border points detected.")

# Run the main function
if __name__ == "__main__":
    main()

