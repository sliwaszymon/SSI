import pandas as pd


class Dataset:
    df: pd.DataFrame

    def __init__(self, file_name):
        self.df = pd.read_csv(file_name, delimiter=r'\s+', header=None)

    def get_value(self, row: int, col: int) -> float | int:
        return self.df[col][row]

    def get_type(self, row: int, col: int) -> str:
        return type(self.get_value(row, col))

    def get_value_and_type(self, row: int, col: int) -> tuple:
        value = self.get_value(row, col)
        return value, type(value)

    def get_full_dataset(self) -> pd.DataFrame:
        return self.df


def main() -> None:
    ds = Dataset('iris.txt')
    print(f'Value: {ds.get_value(0,2)}, Type: {ds.get_type(0,2)}')
    print(f'Value: {ds.get_value(1, 4)}, Type: {ds.get_type(1, 4)}')
    print(f'{ds.get_value_and_type(20, 4)}')
    print(ds.get_full_dataset())


if __name__ == '__main__':
    main()
