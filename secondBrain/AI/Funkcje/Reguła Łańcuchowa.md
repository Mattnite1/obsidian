```python
from numpy import ndarray
from typing import List, Callable
import numpy as np

Array_Function = Callable[[ndarray], ndarray] #przypisanie do aliasu (alternatywnej nazwy) funkcji przyjmujacej ndarray 
Chain = List[Array_Function] #zdefiniowanie naszej listy

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    return (func(input_ + delta) - func(input_ - delta) / (2*delta)) #funkcja obliczajaca pochodna funkcji

def sigmoid(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x)) #funkcja, która kazda liczbe z zakresu (-∞; +∞) sprowadza do liczby z zakresu (-1; 1)

def square(x: ndarray) -> ndarray:
    return x ** 2

def chain_deriv_2(chain: Chain, input_range: ndarray) -> ndarray:
    assert len(chain) == 2 #sprawdzamy czy dlugosc listy jest 2 
    assert input_range.ndim == 1 #sprawdzany czy tablica jest jednowymiarowa

    f1 = chain[0] #przypisuje do f1 pierwsza funkcje z listy
    f2 = chain[1]

    f1_of_x = f1(input_range)

    df1dx = deriv(f1, input_range) #przypisuje do df1dx wynik pochodnej 
    df2du = deriv(f2, f1(input_range))

    return df1dx * df2du #mnozy wyniki dwoch pochodnych funkcji 

function_chain = [sigmoid, square] #przypisuje funkcje do naszej listy
a = np.array([1,2,3])

print(chain_deriv_2(function_chain, a))
```


