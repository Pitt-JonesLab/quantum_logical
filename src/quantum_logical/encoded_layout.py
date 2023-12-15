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


class EncodedLayout(Layout):
    """Subclass Layout to make an EncodedLayout object.

    Rather than mapping virtual to physical bits, map logical virtual
    bits to virtual quantum and classical registers.
    """

    def __init__(self, input_dict=None):
        """Initialize an EncodedLayout object.

        Args:
            input_dict (dict): A dictionary representing a layout.
        """
        super().__init__(input_dict)

    def from_dict(self, input_dict):
        """Populate a Layout from a dictionary.

        The dictionary must be a bijective mapping between
        virtual qubits (tuple) and virtual registers
        (tuple of QuantumRegister and ClassicalRegister)

        Args:
            input_dict (dict):
                e.g.::

                {(QuantumRegister(3, 'qr'), 0):
                    (QuantumRegister(3, "er0"), ClassicalRegister(3, "cr0")),
                 (QuantumRegister(3, 'qr'), 1):
                    (QuantumRegister(3, "er1"), ClassicalRegister(3, "cr1")),
                 (QuantumRegister(3, 'qr'), 2):
                    (QuantumRegister(3, "er2"), ClassicalRegister(3, "cr2"))}]

                Can be written more concisely as follows:

                * logical to encoded virtual qubits:
                    {qr[0]: (er0, cr0), qr[1]: (er1, cr1), qr[2]: (er2, cr2)}

                * encoded to logical virtual qubits:
                    {(er0, cr0): qr[0], (er1, cr1): qr[1], (er2, cr2): qr[2]}
        """
        # NOTE: this super class relies on Layout.order_based_on_type
        # which is overridden below
        return super().from_dict(input_dict)

    @staticmethod
    def order_based_on_type(value1, value2):
        """Decide which one is logical/encoded based on the type.

        Returns (logical, encoded)
        """
        if isinstance(value1, (Qubit, type(None))):
            if (
                isinstance(value2, tuple)
                and len(value2) == 2
                and isinstance(value2[0], QuantumRegister)
                and isinstance(value2[1], ClassicalRegister)
            ):
                logical = value1
                encoded = value2
            else:
                raise LayoutError(
                    "Encoded value must be a tuple of (QuantumRegister, ClassicalRegister)"
                )
        elif isinstance(value2, (Qubit, type(None))):
            if (
                isinstance(value1, tuple)
                and len(value1) == 2
                and isinstance(value1[0], QuantumRegister)
                and isinstance(value1[1], ClassicalRegister)
            ):
                logical = value2
                encoded = value1
            else:
                raise LayoutError(
                    "Encoded value must be a tuple of (QuantumRegister, ClassicalRegister)"
                )
        else:
            raise LayoutError(
                "The map (%s -> %s) must be a (Qubit/tuple -> tuple/Qubit)"
                " or the other way around." % (type(value1), type(value2))
            )
        return logical, encoded

    def __delitem__(self, key):
        """Remove an element from the layout."""
        # if isinstance(key, int):
        #     del self._v2p[self._p2v[key]]
        #     del self._p2v[key]
        if isinstance(key, Qubit):
            del self._p2v[self._v2p[key]]
            del self._v2p[key]
        else:
            raise LayoutError(
                "The key to remove should be of the form"
                " Qubit and %s was provided" % (type(key),)
            )

    def add(self, logical_qubit, encoded_tuple=None):
        """Add a map element between `logical_qubit` and `encoded_tuple`.

        `encoded_tuple` is a pair of QuantumRegister and ClassicalRegister.

        Args:
            logical_qubit (Qubit): A logical qubit.
            encoded_tuple (tuple): A tuple of (QuantumRegister, ClassicalRegister).
        """
        if encoded_tuple is None:
            raise ValueError(
                "encoded_tuple must be provided and be a tuple of (QuantumRegister, ClassicalRegister)"
            )

        if not (
            isinstance(encoded_tuple, tuple)
            and len(encoded_tuple) == 2
            and isinstance(encoded_tuple[0], QuantumRegister)
            and isinstance(encoded_tuple[1], ClassicalRegister)
        ):
            raise ValueError(
                "encoded_tuple must be a tuple of (QuantumRegister, ClassicalRegister)"
            )

        self[logical_qubit] = encoded_tuple

    def add_register(self, quantum_reg, classical_reg):
        """Add a map element between `quantum_reg` and `classical_reg`.

        Add mappings for each qubit in the quantum register to a
        corresponding tuple of quantum and classical registers.

        Args:
            quantum_reg (QuantumRegister): A quantum register.
            classical_reg (ClassicalRegister): A classical register, corresponding to `quantum_reg`.
        """
        raise NotImplementedError
        # if len(quantum_reg) != len(classical_reg):
        #     raise ValueError("Quantum and Classical registers must be of the same size")

        # for qubit, cbit in zip(quantum_reg, classical_reg):
        #     self.add(qubit, (quantum_reg, classical_reg))

    def get_logical_bits(self):
        """Return the dictionary of mapping.

        where the keys are (virtual) logical qubits and the values are
        (virtual) encoded qubits.
        """
        return super().get_virtual_bits()

    def get_encoded_bits(self):
        """Return the dictionary of mapping.

        where the keys are (virtual) encoded qubits and the values are
        (virtual) logical qubits.
        """
        return super().get_physical_bits()
