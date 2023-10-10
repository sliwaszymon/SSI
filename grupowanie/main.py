
# Zad 1.
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
                    print(f'Error: {e}')
        ans.append(part_ans)

    for idx in range(len(ans)-1):
        if len(ans[idx]) != len(ans[idx+1]):
            print('Error: Rows has different length!')
            break

    return ans


def main() -> None:
    # Zad 1.
    probki_str = [["1", "a", "2.2"], ["3", "4", "5"]]
    numery_atr = [0, 2]
    probki_num = probki_str_na_liczby(probki_str, numery_atr)
    print(probki_num)


if __name__ == '__main__':
    main()
