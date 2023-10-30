import numpy as np
import matplotlib.pyplot as plt


class GreedyPointMatching:
    main_bitmap: list[list[int]]
    test_bitmaps: dict
    similarity_measures: dict

    def __init__(self, bitmap: list[list[int]]) -> None:
        self.main_bitmap = bitmap
        self.test_bitmaps = {}
        self.similarity_measures = {}

    def add_test_bitmap(self, new_bitmap: list[list[int]], label: str) -> None:
        self.test_bitmaps[label] = new_bitmap
        self.similarity_measures[label] = -np.inf

    @staticmethod
    def iterate(table: list[list[int]]) -> list[tuple]:
        return [(row_idx, col_idx) for row_idx, row in enumerate(table) for col_idx, _ in enumerate(row)]

    @staticmethod
    def manhattan(p1: list[float | int], p2: list[float | int]) -> float:
        if len(p1) != len(p2):
            raise ValueError("Points must have exact same number of dimensions!")
        return sum([abs(x1 - x2) for x1, x2 in zip(p1, p2)])

    @staticmethod
    def euclidean_distance(p1: list[float | int], p2: list[float | int]) -> float:
        if len(p1) != len(p2):
            raise ValueError("Points must have exact same number of dimensions!")
        return sum([(x1 - x2) ** 2 for x1, x2 in zip(p1, p2)]) ** .5

    def dissimilarity_measures(self, bit_a: list[list[int]], bit_b: list[list[int]]) -> float:
        miara = 0
        for pay, pax in self.iterate(bit_a):
            if bit_a[pay][pax] == 1:
                odl_min = np.inf
                for pby, pbx in self.iterate(bit_b):
                    if bit_b[pby][pbx] == 1:
                        odl_akt = self.euclidean_distance([pax, pay], [pbx, pby])
                        odl_min = min(odl_min, odl_akt)
                miara += odl_min
        return miara

    def two_sided_similarity_measures(self, bit_a: list[list[int]], bit_b: list[list[int]]) -> float:
        return -(self.dissimilarity_measures(bit_a, bit_b) + self.dissimilarity_measures(bit_b, bit_a))

    def calculate_similarities(self) -> None:
        for label, test_bitmap in self.test_bitmaps.items():
            self.similarity_measures[label] = self.two_sided_similarity_measures(self.main_bitmap, test_bitmap)

    def visualize(self):
        fig, axes = plt.subplots(nrows=2, ncols=len(self.test_bitmaps.keys()))
        axes[0, 0].imshow(self.main_bitmap, cmap='Greys')
        axes[0, 0].set_title('Obraz testowy')

        for x in range(1, len(self.test_bitmaps.keys())):
            axes[0, x].set_axis_off()

        for idx, kv in enumerate(self.test_bitmaps.items()):
            axes[1, idx].imshow(kv[1], cmap='Grays')
            axes[1, idx].set_title(kv[0])
        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(nrows=1, ncols=2)
        axes[0].imshow(self.main_bitmap, cmap='Greys')
        axes[0].set_title('Obraz testowy')

        l, v = [(kv[0], kv[1]) for kv in self.similarity_measures.items() if kv[1] == max(self.similarity_measures.
                                                                                          values())][0]

        axes[1].imshow(self.test_bitmaps[l], cmap='Greys')
        axes[1].set_title(f'Najbliższy wzorzec {l} o współczynniku {v}')
        plt.tight_layout()
        plt.show()

    def __call__(self, visualize: bool = True) -> None:
        self.calculate_similarities()
        if visualize and len(self.test_bitmaps) > 1:
            self.visualize()
        else:
            print('There must be more test bitmaps otherwise if you have one you know the best one.')


class HopfieldNeuronNetwork:
    weights: np.ndarray
    n: int

    def __init__(self, n: int) -> None:
        self.n = n
        self.weights = np.zeros((n*n, n*n))

    def train(self, train_pattern: np.ndarray) -> None:
        reshaped_train_pattern = train_pattern.reshape(-1, 1)
        self.weights += np.dot(reshaped_train_pattern, reshaped_train_pattern.T)
        np.fill_diagonal(self.weights, 0)

    def denoise_pattern(self, pattern: np.ndarray) -> np.ndarray:
        flat_pattern = pattern.flatten()
        for i in range(self.n ** 2):
            suma = 0
            for j in range(self.n ** 2):
                if i != j:
                    suma += self.weights[i, j] * flat_pattern[j]
            flat_pattern[i] = 1 if suma >= 0 else -1
        return flat_pattern.reshape(self.n, self.n)

    def __call__(self, pattern: np.ndarray, visualize: bool = False) -> None:
        denoised: np.ndarray = self.denoise_pattern(pattern)

        if visualize:
            plt.imshow(denoised, cmap='Grays')
            plt.title('Naprawiony obraz')
            plt.show()
        else:
            print('Repaired matrix:', denoised)


def zad1() -> None:
    test1 = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    test2 = [[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]]
    test3 = [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    wzorzec1 = [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
    wzorzec2 = [[0, 1, 1, 1], [1, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]
    wzorzec3 = [[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0]]

    gpm1 = GreedyPointMatching(test1)
    gpm1.add_test_bitmap(wzorzec1, 'Wzorzec1')
    gpm1.add_test_bitmap(wzorzec2, 'Wzorzec2')
    gpm1.add_test_bitmap(wzorzec3, 'Wzorzec3')
    gpm1()

    gpm2 = GreedyPointMatching(test2)
    gpm2.add_test_bitmap(wzorzec1, 'Wzorzec1')
    gpm2.add_test_bitmap(wzorzec2, 'Wzorzec2')
    gpm2.add_test_bitmap(wzorzec3, 'Wzorzec3')
    gpm2()

    gpm3 = GreedyPointMatching(test3)
    gpm3.add_test_bitmap(wzorzec1, 'Wzorzec1')
    gpm3.add_test_bitmap(wzorzec2, 'Wzorzec2')
    gpm3.add_test_bitmap(wzorzec3, 'Wzorzec3')
    gpm3()


def zad2() -> None:
    patterns = [
        np.array([[1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1]]),
        np.array([[1, -1, -1, -1, 1], [-1, 1, -1, 1, -1], [-1, -1, 1, -1, -1], [-1, 1, -1, 1, -1], [1, -1, -1, -1, 1]]),
        np.array([[-1, -1, 1, -1, -1], [-1, -1, 1, -1, -1], [1, 1, 1, 1, 1], [-1, -1, 1, -1, -1], [-1, -1, 1, -1, -1]])
    ]

    tests = [
        np.array([[-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1], [-1, 1, -1, -1, -1]]),
        np.array([[1, 1, -1, -1, 1], [-1, 1, -1, 1, -1], [-1, 1, 1, 1, -1], [-1, 1, -1, 1, -1], [1, 1, -1, -1, 1]]),
        np.array([[-1, -1, 1, -1, -1], [-1, -1, 1, -1, -1], [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1], [-1, -1, 1, -1, -1]]),
        np.array([[-1, 1, 1, 1, 1], [1, -1, 1, 1, 1], [1, -1, 1, 1, 1], [1, -1, 1, 1, 1], [1, -1, 1, 1, 1]])
    ]

    hnn = HopfieldNeuronNetwork(5)
    for pattern in patterns:
        hnn.train(pattern)


    hnn(tests[0], visualize=True)
    hnn(tests[1], visualize=True)
    hnn(tests[2], visualize=True)
    hnn(tests[3], visualize=True)


def main() -> None:
    # zad1()  # zachłanne dopasowywanie punktów
    zad2()  # sieci hopfielda


if __name__ == '__main__':
    main()
