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


def k_means(data: list[list[float]], k: int, max_iterations: int = 100, visualize: bool = False) -> tuple:
    data = np.array(data)
    # Randomizowanie punktów centralnych
    initial_indices = np.random.choice(range(len(data)), k, replace=False)
    centroids, cluster_assignments = data[initial_indices], None

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


def zad1() -> None:
    probki_str = [["1", "a", "2.2"], ["3", "4", "5"]]
    numery_atr = [0, 2]
    probki_num = probki_str_na_liczby(probki_str, numery_atr)
    print(probki_num)


def zad2() -> None:
    data = read_data('spiralka.txt')
    data = probki_str_na_liczby(data, [0, 1])
    _, _ = k_means(data, 4, visualize=True)


def main() -> None:
    # zad1()
    zad2()


if __name__ == '__main__':
    main()
