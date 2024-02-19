import numpy as np
import abc as ABC
#from quantum_logical.scqubits.modes import Modes
import scqubits as scq

class System:
    def __init__(self, **kwargs):
        self, modes = list[Modes]
    def create_space(modes):
        scq.hilbertspace(modes)
