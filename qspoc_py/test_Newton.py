from . import propagation_method
import numpy as np
H = [[1,-1j],[1j,0]]
psi = np.array([[1],[1]]) / np.sqrt(2)
psi_T = propagation_method.Newton()