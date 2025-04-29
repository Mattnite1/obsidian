Przykład funkcji zagnieżdżonej w pythonie:

```python
Array_Function = Callable[[np.ndarray], np.ndarray]
Chain = List[Array_Function]

def add_ten(arr: np.ndarray) -> np.ndarray:
    return arr + 10

def multiply_by_two(arr: np.ndarray) -> np.ndarray:
    return arr * 2

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    assert len(chain) == 2
    f1 = chain[0]
    f2 = chain[1]

    return (f2(f1(a)))

function_chain = [add_ten, multiply_by_two]

print(chain_length_2(function_chain, a))
```

Wyjaśnienie:
	List - typ kolekcji. Może zawierać różne typy danych
	Callable -  obiekt, który można wywołać jak funkcję. 
	Chain - reprezentuje listę funkcji   