import numpy as np
import matplotlib.pyplot as plt


def probki_str_na_liczby(probki_str: list[list[str]], numery_atr: list[int]) -> list[list[float]]:
    ans = []

    for row in probki_str:
        part_ans = []
        for idx, col in enumerate(row):
            if idx in numery_atr:
                try:
                    parsed = float(col)
                    part_ans.append(parsed)
                except ValueError as e:
                    # obsługa wyjątki poczas próby parsowania ze str na float jeżeli col nie jest liczbą
                    print(f'Error: {e}')
        ans.append(part_ans)

    for idx in range(len(ans)-1):
        # obsługa wyjątku różnicy długości wierszy macierzy danych
        if len(ans[idx]) != len(ans[idx+1]):
            print('Error: Rows has different length!')
            break

    return ans


def read_data(filename: str) -> list[list[str]]:
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(line.split())
    return data


def initialize_centroids(data: np.ndarray, k: int) -> np.ndarray:
    # Randomowo wybiera k wierszy z data
    initial_indices = np.random.choice(range(len(data)), k, replace=False)
    centroids = data[initial_indices]
    return centroids


def k_means(data: list[list[float]], k: int, max_iterations: int = 100,
            visualize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data = np.array(data)

    centroids, cluster_assignments = initialize_centroids(data, k), None

    for _ in range(max_iterations):
        # Przypisanie danych do najbliższych grup wg. punktów centralnych
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

        # Nowe punkty centralne
        new_centroids = np.array([data[cluster_assignments == k].mean(axis=0) for k in range(k)])

        # sprawdzanie zbieżności starych punktów centralnych z nowymi (optymalizacja)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    if visualize:
        for x in range(k):
            plt.scatter(data[cluster_assignments == x][:, 0], data[cluster_assignments == x][:, 1],
                        label=f'Cluster {x + 1}')

        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.legend()
        plt.show()

    return cluster_assignments, centroids


def assign_clusters(data: np.ndarray, centroids: np.ndarray, m: int) -> np.ndarray:
    num_clusters = len(centroids)
    num_points = len(data)

    # Initialize the membership matrix
    membership = np.zeros((num_points, num_clusters))

    for i in range(num_points):
        for j in range(num_clusters):
            # Calculate the membership of data point i to cluster j
            num = np.linalg.norm(data[i] - centroids[j])

            # Handle division by zero
            if num == 0:
                membership[i][j] = 1
            else:
                denom = sum(
                    np.power(num / max(np.linalg.norm(data[i] - centroids[c]), 1e-10), 2 / (m - 1)) for c in
                    range(num_clusters))
                membership[i][j] = 1 / denom

    return membership


def update_centroids(data: np.ndarray, membership: np.ndarray, m: int) -> np.ndarray:
    num_clusters = membership.shape[1]
    num_points = membership.shape[0]

    new_centroids = np.zeros((num_clusters, data.shape[1]))

    for j in range(num_clusters):
        numerator = sum(np.power(membership[i][j], m) * data[i] for i in range(num_points))
        denominator = sum(np.power(membership[i][j], m) for i in range(num_points))
        new_centroids[j] = numerator / denominator

    return new_centroids


def fuzzy_c_means(data, k: int, m: int, max_iterations: int = 100, tolerance: float = 1e-4, visualize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    data = np.array(data)
    centroids, membership = initialize_centroids(data, k), None

    for _ in range(max_iterations):
        membership = assign_clusters(data, centroids, m)
        new_centroids = update_centroids(data, membership, m)

        # sprawdzanie zbieżności starych punktów centralnych z nowymi (optymalizacja)
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break

        centroids = new_centroids

    if visualize:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        plt.figure(figsize=(8, 6))

        for i in range(len(data)):
            cluster = np.argmax(membership[i])
            plt.scatter(data[i][0], data[i][1], c=colors[cluster], marker='o')

        for i, centroid in enumerate(centroids):
            cluster_color = colors[i]
            plt.scatter(centroid[0], centroid[1], c=cluster_color, marker='x', label=f'Centroid {i + 1}')

        plt.title('Fuzzy C-Means Clustering')
        plt.legend()
        plt.show()

    return centroids, membership


def zad1() -> None:
    probki_str = [["1", "a", "2.2"], ["3", "4", "5"]]
    numery_atr = [0, 2]
    probki_num = probki_str_na_liczby(probki_str, numery_atr)
    print(probki_num)


def zad2() -> None:
    data = read_data('spiralka.txt')
    data = probki_str_na_liczby(data, [0, 1])
    cluster_assignments, centroids = k_means(data, 4, visualize=True)
    print("Ostateczne punkty centralne:\n", centroids)


def zad3() -> None:
    data = read_data('spiralka.txt')
    data = probki_str_na_liczby(data, [0, 1])

    centroids, membership = fuzzy_c_means(data, 3, 2, visualize=True)
    print("Ostateczne punkty centralne:\n", centroids)


def main() -> None:
    zad1() # wczytywanie próbek
    zad2() # k-means
    zad3() # fuzzy_c_means


if __name__ == '__main__':
    np.random.seed(0)
    main()
