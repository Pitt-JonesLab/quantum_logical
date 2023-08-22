import ipywidgets as widgets
import numpy as np
import qutip

# import matplotlib.pyplot as plt
# import rustworkx as rx
# from rustworkx.visualization import mpl_draw
from qiskit.circuit import Gate, Parameter, QuantumCircuit


class Hamiltonian:
    def __init__(self, H):
        """Initialize the Hamiltonian.

        Args:
            H: A function that returns the Hamiltonian matrix.
        """
        self.H = H

    def construct_U(self, t) -> qutip.Qobj:
        """Construct the unitary evolution operator for time t.

        Args:
            t: Time.

        Returns:
            Unitary operator for the given time.
        """
        return (-1j * t * self.H).expm()

    @property
    def unitary(self, t):
        return self.construct_U(t)


class Interaction:
    def construct_U(self, t) -> qutip.Qobj:
        """Construct the unitary evolution operator for time t.

        Args:
            t: Time.

        Returns:
            Unitary operator for the given time.
        """
        return self.construct_H().construct_U(t)


class QubitInteraction(Interaction):
    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0):
        """Initialize the qubit interaction parameters.

        Args:
            gc: Coupling strength.
            gg: Gate strength.
            phi_c: Phase for coupling.
            phi_g: Phase for gate.
        """
        self.set_parameters(gc, gg, phi_c, phi_g)

    def set_parameters(self, gc, gg, phi_c=0.0, phi_g=0.0):
        """Set the qubit interaction parameters.

        Args:
            gc: Coupling strength.
            gg: Gate strength.
            phi_c: Phase for coupling.
            phi_g: Phase for gate.
        """
        # Convert to Parameter objects if input is string, else use as-is
        self.gc = Parameter(gc) if isinstance(gc, str) else gc
        self.gg = Parameter(gg) if isinstance(gg, str) else gg
        self.phi_c = Parameter(phi_c) if isinstance(phi_c, str) else phi_c
        self.phi_g = Parameter(phi_g) if isinstance(phi_g, str) else phi_g

    def construct_H(self, A=None, B=None) -> Hamiltonian:
        """Construct the Hamiltonian for the interaction.

        Args:
            A, B: Operator terms for the Hamiltonian.

        Returns:
            The constructed Hamiltonian.
        """
        a = qutip.operators.create(N=2)
        if A is None:
            A = qutip.tensor(a, qutip.operators.identity(2))
        if B is None:
            B = qutip.tensor(qutip.operators.identity(2), a)

        H_c = (
            np.exp(1j * self.phi_c) * A * B.dag()
            + np.exp(-1j * self.phi_c) * A.dag() * B
        )
        H_g = (
            np.exp(1j * self.phi_g) * A * B
            + np.exp(-1j * self.phi_g) * A.dag() * B.dag()
        )
        return Hamiltonian(self.gc * H_c + self.gg * H_g)

    def create_slider_interaction(self):
        """Create interactive sliders for the interaction parameters.

        Returns:
            An interactive widget with sliders.
        """
        gc_slider = widgets.FloatSlider(
            value=self.gc, min=0, max=2 * np.pi, step=np.pi / 16, description="gc:"
        )
        gg_slider = widgets.FloatSlider(
            value=self.gg, min=0, max=2 * np.pi, step=np.pi / 16, description="gg:"
        )
        return widgets.interactive(self.set_parameters, gc=gc_slider, gg=gg_slider)


class QubitSystem(Interaction):
    def __init__(self, qubit_interactions, num_qubits):
        # Same initialization code as before
        self.num_qubits = num_qubits
        self.qubit_interactions = qubit_interactions

        self._params = []
        self.parameter_mapping = []  # List to keep track of the mapping
        for qi, _ in qubit_interactions:
            # Check and add gc parameter
            if isinstance(qi.gc, Parameter):
                self.parameter_mapping.append((qi, "gc"))
                self._params.append(qi.gc)
            # Check and add gg parameter
            if isinstance(qi.gg, Parameter):
                self.parameter_mapping.append((qi, "gg"))
                self._params.append(qi.gg)
            # Check and add phi_c parameter
            if isinstance(qi.phi_c, Parameter):
                self.parameter_mapping.append((qi, "phi_c"))
                self._params.append(qi.phi_c)
            # Check and add phi_g parameter
            if isinstance(qi.phi_g, Parameter):
                self.parameter_mapping.append((qi, "phi_g"))
                self._params.append(qi.phi_g)

    def update_parameters(self, param_values):
        # Iterate through the parameter values and mapping
        for value, (qi, attr_name) in zip(param_values, self.parameter_mapping):
            # Update the corresponding attribute in the Qubit interaction
            setattr(qi, attr_name, value)

    def construct_H(self) -> Hamiltonian:
        """Construct the system Hamiltonian matrix.

        Returns:
            Hamiltonian matrix for the system.
        """
        H = 0
        for qubit_interaction, qubit_pair in self.qubit_interactions:
            H += self.construct_interaction(qubit_interaction, qubit_pair).H
        return Hamiltonian(H)

    def construct_interaction(self, qubit_interaction, qubit_pair):
        a = qutip.operators.create(N=2)

        operators = [qutip.operators.identity(2) for _ in range(self.num_qubits)]
        operators[qubit_pair[0]] = a
        A = qutip.tensor(*operators)

        operators = [qutip.operators.identity(2) for _ in range(self.num_qubits)]
        operators[qubit_pair[1]] = a
        B = qutip.tensor(*operators)

        H_int = qubit_interaction.construct_H(A, B)
        return H_int

    # def display(self):
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    #     # Displaying sliders (QubitInteraction)
    #     for interaction, _ in self.qubit_interactions:
    #         display(interaction.create_slider_interaction())

    #     # Qubit interactions graph
    #     G = rx.PyGraph()
    #     G.add_nodes_from(list(range(self.num_qubits)))
    #     for idx, (interaction, (q1, q2)) in enumerate(self.qubit_interactions):
    #         G.add_edge(q1, q2, interaction.gc)  # or other value for annotation
    #         axes[1].text(
    #             (q1 + q2) / 2, idx, f"{interaction.gc:.2f},{interaction.gg:.2f}"
    #         )

    #     mpl_draw(G, ax=axes[1], with_labels=True)

    #     # # Final computed unitary
    #     # U = self._construct_U_lambda()(t=1.0)  # Assuming U is computed this way
    #     # im = axes[2].imshow(np.abs(U), cmap="viridis")
    #     # fig.colorbar(im, ax=axes[2])

    #     plt.show()


class ParameterizedUnitaryGate(Gate):
    def __init__(self, qubit_system: QubitSystem):
        self.qubit_system = qubit_system
        super().__init__("CG", qubit_system.num_qubits, qubit_system._params)

    def __array__(self, dtype=None):
        # kind of hacky because we don't propagate changes to params until we ask for the array
        # use self.params to update the qubit system
        self.qubit_system.update_parameters(self.params)
        return self.qubit_system.construct_U(t=1.0).full()

    def _define(self):
        self.qubit_system.update_parameters(self.params)
        qc = QuantumCircuit(self.num_qubits)
        U = self.qubit_system.construct_U(t=1.0).full()
        qc.unitary(U, range(self.num_qubits))
        self.definition = qc
