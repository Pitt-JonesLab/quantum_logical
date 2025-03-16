from qutip import tensor, basis


def state(qubit_choices, dim, alpha, beta):
    """Builds state in an arbitrary superposition of |---> and |+++>

        The set of choices decided upon for the state will determine which 
        part of the state is multiplied by which coefficient 
        i.e. state(qubit_choices = ["+", "+", "+"], dim=3, alpha, beta), then 
        the output state = alpha |+++> + beta |--->
        """
    # dictionary of possible states(phase protected).
    if dim == 3:
        state_choices = {"+": (basis(dim, 0) + basis(dim, 2)).unit(),
                        "-": (basis(dim, 0) - basis(dim, 2)).unit(),
                        "1": basis(dim, 1)}
        individual_states = [basis(dim, 0) for _ in range(len(qubit_choices))]
    elif dim == 2:
        state_choices = {"+": (basis(dim, 0) + basis(dim, 1)).unit(),
                        "-": (basis(dim, 0) - basis(dim, 1)).unit()}
        individual_states = [basis(dim, 0) for _ in range(len(qubit_choices))]
    
    # building the superposition
    qubit_choices_superposition = [0 for _ in range(len(qubit_choices))]
    for i in range(len(qubit_choices)):
        if qubit_choices[i] == "+":
            qubit_choices_superposition[i] = "-"
        elif qubit_choices[i] == "-":
            qubit_choices_superposition[i] = "+"
        else:
            qubit_choices_superposition[i] = qubit_choices[i]

    individual_states_superposition = [basis(dim, 0)
                                       for _ in range(len(qubit_choices_superposition))]

    # building the constituent state vectors
    for i in range(len(qubit_choices)):
        individual_states[i] = state_choices[qubit_choices[i]]
        individual_states_superposition[i] = state_choices[qubit_choices_superposition[i]]

    state_vec = (alpha * tensor(individual_states) +
             beta * tensor(individual_states_superposition)).unit()  # state vector

    rho_state = state_vec * state_vec.dag()

    return rho_state, state_vec


def combined_state():

    states = [[["-", "-", "-"], 1, 0], [["+", "+", "+"], 1, 0],
              [["+", "+", "+"], 1, 1], [["+", "+", "+"], 1, -1],
              [["+", "+", "+"], 1, 1j], [["+", "+", "+"], 1, -1j]]

    states_created = [state(qubit_choices=states[0], dim=3, alpha=states[1],
                            beta=states[2])[0] for states in states]

    state_final = sum(states_created) / sum(states_created).tr()

    return state_final
