"""Subclass Layout to make an EncodedLayout object.

rather than mapping virtual to physical bits, we need to map logical
virtual to encoded virtual bits. the reason we need to subclass it is
because the virtual bits can be a list (e.g. repetition code) the
logical qubit 0 is now encoded qubits [0,1,2]. we might find it useful
to have logical qubits point to both a quantum register and a classical
register (where the syndrome measurements take place).
"""
from qiskit import ClassicalRegister, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.transpiler.exceptions import LayoutError
from qiskit.transpiler.layout import Layout


class EncodedRegisters:
    """EncodedRegisters is a set of quantum and classical registers."""

    def __init__(
        self,
        codeword_register,
        ancilla_register,
        classical_codeword_register,
        classical_ancilla_register,
    ):
        """Initialize an EncodedRegisters object."""
        self.codeword_register = codeword_register
        self.ancilla_register = ancilla_register
        self.classical_codeword_register = classical_codeword_register
        self.classical_ancilla_register = classical_ancilla_register

    def __iter__(self):
        """Define list of registers to iterate over."""
        return iter(
            [
                self.codeword_register,
                self.ancilla_register,
                self.classical_codeword_register,
                self.classical_ancilla_register,
            ]
        )

    def __hash__(self):
        """Define hash function."""
        return hash(
            (
                self.codeword_register,
                self.ancilla_register,
                self.classical_codeword_register,
                self.classical_ancilla_register,
            )
        )

    def __eq__(self, other):
        """Define equality function."""
        return (
            self.codeword_register == other.codeword_register
            and self.ancilla_register == other.ancilla_register
            and self.classical_codeword_register == other.classical_codeword_register
            and self.classical_ancilla_register == other.classical_ancilla_register
        )


class EncodedLayout(Layout):
    """EncodedLayout maps qubits to sets of quantum/classical registers.

    This subclass of Layout is designed to handle the mapping of logical
    virtual qubits to virtual quantum and classical registers,
    specifically for encoded quantum circuits. Each logical qubit is
    associated with a set of registers including a codeword quantum
    register, an ancilla quantum register, and their corresponding
    classical registers.
    """

    def __init__(self, input_dict=None):
        """Initialize an EncodedLayout object."""
        super().__init__(input_dict)

    def from_dict(self, input_dict):
        """Populate the EncodedLayout from a dictionary mapping.

        Args:
            input_dict (dict): Mapping from logical qubits (Qubit) to dictionaries of registers.
        """
        # return super().from_dict(input_dict)
        for key, value in input_dict.items():
            virtual, physical = EncodedLayout.order_based_on_type(key, value)
            self._p2v[physical] = virtual
            if virtual is None:
                continue
            self._v2p[virtual] = physical

    def __setitem__(self, key, value):
        """Set a mapping in the layout."""
        virtual, physical = EncodedLayout.order_based_on_type(key, value)
        self._set_type_checked_item(virtual, physical)

    @staticmethod
    def order_based_on_type(value1, value2):
        """Determine which value is k/v based on their types.

        Returns:
            EncodedRegisters: A EncodedRegisters object.
        """
        logical, encoded = None, None
        if isinstance(value1, Qubit):
            logical = value1
            encoded = value2
        elif isinstance(value2, Qubit):
            logical = value2
            encoded = value1
        else:
            raise LayoutError("One of the values must be a Qubit.")

        if not isinstance(encoded, EncodedRegisters):
            raise LayoutError("Encoded value must be a list.")

        for reg in encoded:
            if not (
                isinstance(reg, QuantumRegister) or isinstance(reg, ClassicalRegister)
            ):
                raise LayoutError(
                    "Encoded value must be a list of QuantumRegisters or ClassicalRegisters."
                )

        return logical, encoded

    def __delitem__(self, key):
        """Remove a mapping from the layout.

        Args:
            key (Qubit): The logical qubit whose mapping is to be removed.
        """
        if isinstance(key, Qubit):
            del self._p2v[self._v2p[key]]
            del self._v2p[key]
        else:
            raise LayoutError("The key to remove must be a Qubit.")

    def add(
        self,
        logical_qubit,
        codeword_register,
        ancilla_register,
        classical_codeword_register,
        classical_ancilla_register,
    ):
        """Map a logical qubit to a set of encoded quantum/classical registers.

        Args:
            logical_qubit (Qubit): A logical qubit.
            codeword_register (QuantumRegister): Quantum register for codeword qubits.
            ancilla_register (QuantumRegister): Quantum register for ancilla qubits.
            classical_codeword_register (ClassicalRegister): Classical register for codeword qubit measurements.
            classical_ancilla_register (ClassicalRegister): Classical register for ancilla qubit measurements.
        """
        if not isinstance(logical_qubit, Qubit):
            raise ValueError("Logical qubit must be a Qubit instance.")

        encoded_registers = EncodedRegisters(
            codeword_register,
            ancilla_register,
            classical_codeword_register,
            classical_ancilla_register,
        )

        for reg in encoded_registers:
            if not (
                isinstance(reg, QuantumRegister) or isinstance(reg, ClassicalRegister)
            ):
                raise ValueError(
                    f"{reg} must be a QuantumRegister or ClassicalRegister."
                )

        self[logical_qubit] = encoded_registers

    def get_logical_to_encoded_mapping(self):
        """Get the mapping of logical qubits to encoded registers.

        Returns:
            dict: A dictionary mapping logical qubits to their corresponding sets of encoded registers.
        """
        return self._v2p

    def get_encoded_to_logical_mapping(self):
        """Get the mapping of encoded registers to logical qubits.

        Returns:
            dict: A dictionary mapping sets of encoded registers to their corresponding logical qubits.
        """
        return self._p2v
