from typing import Tuple, Callable

import numpy as np
from sklearn.linear_model import LinearRegression


def _array_create(n, m):
    return [0 if i < m else i - m + 1 for i in range(0, n)]


def _possibility(n, m):
    v = _array_create(n, m)
    same, not_same = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if v[i] == v[j]:
                same += 1
            else:
                not_same += 1

    return same / (not_same + same)


def _get_m_for_n(n, p=0.6):
    left, right = 0, n
    while left < right:
        middle = (left + right) // 2
        if _possibility(n, middle) < p:
            left = middle + 1
        else:
            right = middle

    return right


def get_reg_for_prob(prob=0.5) -> Tuple[Callable[[int], int], Callable[[int], int]]:
    x_arr = np.asarray(range(2, 150))
    y_arr = np.asarray([_get_m_for_n(i, p=prob) for i in x_arr])

    x_to_y = LinearRegression()
    x_to_y.fit(x_arr.reshape(-1, 1), y_arr)

    def _x_to_y(x: int) -> int:
        raw = x_to_y.predict(np.asarray([[x]]))[0].tolist()
        return int(round(raw))

    y_to_x = LinearRegression()
    y_to_x.fit(y_arr.reshape(-1, 1), x_arr)

    def _y_to_x(y: int) -> int:
        raw = y_to_x.predict(np.asarray([[y]]))[0].tolist()
        return int(round(raw))

    return _x_to_y, _y_to_x
