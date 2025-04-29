
```python
from numpy import ndarray
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt 

Array_Function = Callable[[ndarray], ndarray] 

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x)) 

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:

    return (func(input_ + delta) - func(input_ - delta)) / (2*delta)

def matrix_function_forward_sum(X: ndarray, W: ndarray, sigma: Array_Function) -> float: 
    assert X.shape[1] == W.shape[0]

    N = np.dot(X, W)
    S = sigma(N)
    L = np.sum(S)
    return L

def matrix_function_backward_sum(X: ndarray, W:ndarray, sigma: Array_Function) -> ndarray:
    assert X.shape[1] == W.shape[0]
    
    N = np.dot(X, W)

    S = sigma(N)

    L = np.sum(S)

    dLdS = np.ones_like(S)
    dSdN = deriv(sigma, N)
    dLdN = dLdS * dSdN

    dNdX = np.transpose(W, (1,0))

    dLdX = np.dot(dSdN, dNdX)

    return dLdX 
    
np.random.seed(190204)
X = np.random.randn(3,3)
W = np.random.randn(3,2)

print("X:", X)
print("L:")
print(matrix_function_forward_sum(X, W, sigmoid))
print()
print("dLdX:")
print(matrix_function_backward_sum(X, W, sigmoid))

X1 = X.copy()
X1[0, 0] += 0.001

def matrix_function_forward_sum(X: ndarray,
                                W: ndarray,
                                sigma: Array_Function,
                                modify_x11: bool = False,
                                x11: float = 0.5) -> float:
    '''
    Computing the result of the forward pass of this function with
    input Tensors X and W and function sigma.
    '''
    assert X.shape[1] == W.shape[0]
    
    X1 = X.copy()
    if modify_x11:
        X1[0][0] = x11

    # matrix multiplication
    N = np.dot(X1, W)

    # feeding the output of the matrix multiplication through sigma
    S = sigma(N)

    # sum all the elements
    L = np.sum(S)

    return L

print(round(
        (matrix_function_forward_sum(X1, W, sigmoid) - \
         matrix_function_forward_sum(X, W, sigmoid)) / 0.001, 4))

print("X:")
print(X)

x11s = np.arange(X[0][0] - 1, X[0][0] + 1, 0.01)
Ls = [matrix_function_forward_sum(X, W, sigmoid, modify_x11=True, x11=x11) for x11 in x11s]

plt.plot(x11s, Ls)
# plt.title("Value of $L$ as $x_{11}$ changes holding\nother values of $X$ and $W$ constant")
plt.xlabel("$x_{11}$")
plt.ylabel("$L$")
plt.show()
# plt.savefig(IMG_FOLDER_PATH + "18_x11_vs_L.png");
```
### **Pełne obliczanie gradientów wraz z:**
**Forward Pass (przejście do przodu):

- Oblicza wyjście funkcji aktywacji (`sigmoid`) po zastosowaniu na przekształconych macierzach `X` i `W`.
- Oblicza sumę wartości wyjściowych po aktywacji, co daje wynik funkcji `L`.

**Backward Pass (propagacja wsteczna)**:

- Oblicza gradient funkcji wyjściowej `L` względem danych wejściowych `X`.
- Gradient ten pozwala określić, jak bardzo zmiany w `X` wpływają na zmianę wartości `L`.
- Wykorzystuje metodę różnic skończonych do przybliżenia pochodnych funkcji aktywacji sigmoid.

### Podsumowanie funkcji kodu

Kod ten mógłby być częścią większego algorytmu treningu sieci neuronowej, gdyż realizuje:

- obliczanie wyjścia warstwy (forward pass),
- przybliżanie pochodnych funkcji aktywacji,
- propagację gradientów (backward pass) — potrzebną do aktualizacji wag w procesie uczenia.