Mnozenie macierzy A przez macierz B polega na przemnozeniu kazdego wierszu macierzy A przed kazda kolumne macierzy B. Przykładowy kod, gdzie:
- tablica a to wiersze
- tablica b to kolumny

```import numpy as np
from numpy import ndarray

def matmul_forward(X: ndarray, W:ndarray) -> ndarray:
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)

    return N 

a = np.array([2,4,5,6,7]).reshape(1, -1) #wiersze
b = np.array([5,6,3,5,1]).reshape(-1, 1) #kolumny

print(matmul_forward(a,b))
```

