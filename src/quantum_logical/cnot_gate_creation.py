import qutip as qt
from qutip import tensor, qeye, Qobj

def gate_expand_2toN(U, N, control=None, target=None, targets=None):
            """
            Create a Qobj representing a two-qubit gate that act on a system with N
            qubits.

            Parameters
            ----------
            U : Qobj
                The two-qubit gate

            N : integer
                The number of qubits in the target space.

            control : integer
                The index of the control qubit.

            target : integer
                The index of the target qubit.

            targets : list
                List of target qubits.

            Returns
            -------
            gate : qobj
                Quantum object representation of N-qubit gate.

            """

            if targets is not None:
                control, target = targets

            if control is None or target is None:
                raise ValueError("Specify value of control and target")

            if N < 2:
                raise ValueError("integer N must be larger or equal to 2")

            if control >= N or target >= N:
                raise ValueError("control and not target must be integer < integer N")

            if control == target:
                raise ValueError("target and not control cannot be equal")

            p = list(range(N))

            if target == 0 and control == 1:
                p[control], p[target] = p[target], p[control]

            elif target == 0:
                p[1], p[target] = p[target], p[1]
                p[1], p[control] = p[control], p[1]

            else:
                p[1], p[target] = p[target], p[1]
                p[0], p[control] = p[control], p[0]
            

            return tensor([U] + [qeye(3)] * (N - 2)).permute(p)

        
def cnot(N=None, control = None, target= None, high= None, low= None):
    """
    Quantum object representing the CNOT gate.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    Examples
    --------
    >>> cnot()
    Quantum object: dims = [[2, 2], [2, 2]], 
shape = [4, 4], type = oper, isHerm = True
    Qobj data =
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
        [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
        [ 0.+0.j  0.+0.j  0.+0.j  1.+0.j]
        [ 0.+0.j  0.+0.j  1.+0.j  0.+0.j]]

    """
    if (control == 1 and target == 0) and N is None:
        N = 2

    if N is not None:
        return gate_expand_2toN(cnot(N=None, high=high, low=low), N, control, target)
    else:
        # return Qobj([[1, 0, 0, 0],
        #             [0, 1, 0, 0],
        #             [0, 0, 0, 1],
        #             [0, 0, 1, 0]],
        #             dims=[[2, 2], [2, 2]])
        if high == 2 and low == 0:
            return Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0]]
                            , dims=[[3, 3], [3, 3]])
        elif high == 1 and low == 0:
            return Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]]
                            , dims=[[3, 3], [3, 3]])
        elif high == 2 and low == 1:
             return Qobj([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0]]
                            , dims=[[3, 3], [3, 3]])
        elif high == 0 and low == 1:
             return Qobj([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]]
                            , dims=[[3, 3], [3, 3]])
        elif high == [2,0] and low == 1:
             return Qobj([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0]]
                            , dims=[[3, 3], [3, 3]])