import numba
from numba import jit, prange, vectorize
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
import pandas as pd
import time
start_time = time.time()

savedfractal = True

dpino = 150  # 300 # 1200 # 300 for high res plotting
figuresize = 1  # 40 # 20
# 6000 # 1000 # figsize=20 x 20 with dpi=300 -> size = 20x300=6000
size = figuresize*dpino
heightsize = size
widthsize = size
power_offset = 0
escapeno = 2

#plt.figure(figsize=(figuresize, figuresize))
#plt.imshow(julia_set_jit(statevector_data=tatevector,max_iterations=max_iterations), cmap='magma')
#plt.axis('off')
#print("--- drawn image %s seconds ---" % (time.time() - start_time))
#plt.show()

@jit(nopython=True, cache=False)
def create_fraction(n_sub_expressions: int, is_numerator: bool, powers: List[int], indices: List[int]):
    if is_numerator:
        powers = powers + [n_sub_expressions + power_offset]
        indices = indices + [(n_sub_expressions - 1) * 2]
    else:
        powers = powers + [n_sub_expressions + power_offset]
        indices = indices + [((n_sub_expressions - 1) * 2) + 1]
    if n_sub_expressions - 1 > 0:
        return create_fraction(n_sub_expressions - 1, is_numerator, powers, indices)
    else:
        return powers, indices

@jit(nopython=True, cache=False)
def create_formula(n_sub_expressions: int):
    upper_pwrs, upper_idxs = create_fraction(n_sub_expressions, is_numerator=True, powers=[
                                                numba.int64(x) for x in range(0)], indices=[numba.int64(x) for x in range(0)])
    lower_pwrs, lower_idxs = create_fraction(n_sub_expressions, is_numerator=False, powers=[
                                                numba.int64(x) for x in range(0)], indices=[numba.int64(x) for x in range(0)])
    return (np.array(upper_pwrs), np.array(upper_idxs), np.array(lower_pwrs), np.array(lower_idxs))

@jit(nopython=True, cache=False, parallel=True, nogil=True)
def julia_set_jit(statevector_data: np.ndarray, size: int, x: int = 0, y: int = 0, zoom: int = 1, max_iterations: int = 100, number_of_qubits: int = 8):
    # Generate the number of expressions, its powers and indexes for the dynamic Julia Set Mating formular
    #######################################################################################################
    # Calculate how many sub expressions are to be produced
    
    height = size
    width = size
    if number_of_qubits == 1:
        n_sub_expressions = number_of_qubits
    else:
        n_sub_expressions = 2 ** number_of_qubits // 2
    # Calculate the powers and indexes
    (upper_pwrs, upper_idxs, lower_pwrs, lower_idxs) = create_formula(n_sub_expressions)

    # To make navigation easier we calculate these values
    x_width: float = 1.5

    x_from: float = x - x_width / zoom
    x_to: float = x + x_width / zoom

    y_height: float = 1.5 * height / width
    y_from: float = y - y_height / zoom
    y_to: float = y + y_height / zoom
    # Here the actual algorithm starts and the z parameter is defined for the Julia set function
    x = np.linspace(x_from, x_to, width).reshape((1, width))
    y = np.linspace(y_from, y_to, height).reshape((height, 1))
    z = (x + 1j * y)

    # To keep track in which iteration the point diverged
    div_time = np.full(z.shape, max_iterations - 1, dtype=numba.uint8)

    # To keep track on which points did not converge so far
    m = np.full(z.shape, True, dtype=numba.bool_)

    for j in range(max_iterations):
        for x in prange(size):
            for y in prange(size):
                if m[x, y]:
                    # Create first sub-equation of Julia mating equation
                    uval = z[x, y] ** (upper_pwrs[0])
                    lval = z[x, y] ** (lower_pwrs[0])
                    # Add middle sub-equation(s) of Julia mating equation
                    for i in prange(n_sub_expressions - 1):
                        uval = uval + \
                            statevector_data[upper_idxs[i]] * z[x, y] ** (upper_pwrs[i + 1])
                        lval = lval + \
                            statevector_data[lower_idxs[i]] * z[x, y] ** (lower_pwrs[i + 1])

                    # Add final sub-equation of Julia mating equation
                    uval = uval + \
                        statevector_data[upper_idxs[n_sub_expressions - 1]]
                    lval = lval + \
                        statevector_data[lower_idxs[n_sub_expressions - 1]]

                    z[x, y] = uval / lval
                    if abs(z[x, y]) > escapeno:
                        m[x, y] = False
                        div_time[x, y] = j
    return div_time


