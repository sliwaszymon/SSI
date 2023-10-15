import numpy as np
import matplotlib.pyplot as plt


class Data:
    data: list[list[float]]

    @staticmethod
    def probki_str_na_liczby(probki_str: list[list[str]], numery_atr: list[int] = None) -> list[list[float]]:
        if numery_atr is None:
            numery_atr = [x for x in range(len(probki_str))]

        ans = []

        for row in probki_str:
            part_ans = []
            for idx, col in enumerate(row):
                if idx in numery_atr:
                    try:
                        parsed = float(col)
                        part_ans.append(parsed)
                    except ValueError as e:
                        print(f'Error: {e}')
            ans.append(part_ans)

        for idx in range(len(ans) - 1):
            if len(ans[idx]) != len(ans[idx + 1]):
                print('Error: Rows has different length!')
                break

        return ans

    def read_data(self, filename: str) -> None:
        data = []
        with open(filename, 'r') as f:
            for line in f:
                data.append(line.split())
        self.data = self.probki_str_na_liczby(data)

    def get_data(self) -> list[list[float]]:
        return self.data


class KMeans:
    data: np.ndarray
    k: int
    max_iterations: int
    centroids: np.ndarray
    membership: np.ndarray

    def __init__(self, data: list[list[float]], k: int, max_iterations: int = 100) -> None:
        self.data = np.array(data)
        self.k = k
        self.max_iterations = max_iterations

    def initialize_centroids(self) -> None:
        initial_indices = np.random.choice(range(len(self.data)), self.k, replace=False)
        self.centroids = self.data[initial_indices]

    def visualize(self) -> None:
        for x in range(self.k):
            plt.scatter(self.data[self.membership == x][:, 0], self.data[self.membership == x][:, 1],
                        label=f'Cluster {x + 1}')

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
        plt.legend()
        plt.show()

    def __call__(self, visualize=False) -> None:

        self.initialize_centroids()

        for _ in range(self.max_iterations):
            distances = np.linalg.norm(self.data[:, np.newaxis] - self.centroids, axis=2)
            self.membership = np.argmin(distances, axis=1)

            new_centroids = np.array([self.data[self.membership == k].mean(axis=0) for k in range(self.k)])
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        if visualize:
            self.visualize()

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def get_membership(self) -> np.ndarray:
        return self.membership


class FuzzyCMeans:
    data: np.ndarray
    k: int
    m: int
    max_iterations: int
    tolerance: float
    centroids: np.ndarray
    membership: np.ndarray

    def __init__(self, data: list[list[float]], k: int,
                 m: int, max_iterations: int = 100, tolerance: float = 1e-4) -> None:
        self.data = np.array(data)
        self.k = k
        self.m = m
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def initialize_centroids(self) -> None:
        initial_indices = np.random.choice(range(len(self.data)), self.k, replace=False)
        self.centroids = self.data[initial_indices]

    def assign_clusters(self) -> None:
        num_clusters = len(self.centroids)
        num_points = len(self.data)

        membership = np.zeros((num_points, num_clusters))

        for i in range(num_points):
            for j in range(num_clusters):
                num = np.linalg.norm(self.data[i] - self.centroids[j])
                if num == 0:
                    membership[i][j] = 1
                else:
                    denom = sum(
                        np.power(num / max(np.linalg.norm(self.data[i] - self.centroids[c]), 1e-10),
                                 2 / (self.m - 1)) for c in range(num_clusters))
                    membership[i][j] = 1 / denom

        self.membership = membership

    def update_centroids(self) -> None:
        num_clusters = self.membership.shape[1]
        num_points = self.membership.shape[0]

        new_centroids = np.zeros((num_clusters, self.data.shape[1]))

        for j in range(num_clusters):
            numerator = sum(np.power(self.membership[i][j], self.m) * self.data[i] for i in range(num_points))
            denominator = sum(np.power(self.membership[i][j], self.m) for i in range(num_points))
            new_centroids[j] = numerator / denominator

        if np.linalg.norm(new_centroids - self.centroids) > self.tolerance:
            self.centroids = new_centroids

    def visualize(self) -> None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        plt.figure(figsize=(8, 6))

        for i in range(len(self.data)):
            cluster = np.argmax(self.membership[i])
            plt.scatter(self.data[i][0], self.data[i][1], c=colors[cluster], marker='o')

        for i, centroid in enumerate(self.centroids):
            cluster_color = colors[i]
            plt.scatter(centroid[0], centroid[1], c=cluster_color, marker='x', label=f'Centroid {i + 1}')

        plt.title('Fuzzy C-Means Clustering')
        plt.legend()
        plt.show()

    def __call__(self, visualize=False) -> None:
        self.initialize_centroids()

        for _ in range(self.max_iterations):
            self.assign_clusters()
            self.update_centroids()

        if visualize:
            self.visualize()

    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def get_membership(self) -> np.ndarray:
        return self.membership


def zad1() -> None:
    probki_str = [["1", "a", "2.2"], ["3", "4", "5"]]
    numery_atr = [0, 2]

    data = Data().probki_str_na_liczby(probki_str, numery_atr)
    print(data)


def zad2() -> None:
    data = Data()
    data.read_data('spiralka.txt')
    data.get_data()

    kmeans = KMeans(data.get_data(), 4)
    kmeans(visualize=True)
    print("Ostateczne punkty centralne:\n", kmeans.get_centroids())


def zad3() -> None:
    data = Data()
    data.read_data('spiralka.txt')
    data.get_data()

    fuzzycm = FuzzyCMeans(data.get_data(), 3, 2)
    fuzzycm(visualize=True)
    print("Ostateczne punkty centralne:\n", fuzzycm.get_centroids())


def main() -> None:
    zad1()  # wczytywanie próbek
    zad2()  # k-means
    zad3()  # fuzzy_c_means


if __name__ == '__main__':
    np.random.seed(0)
    main()
