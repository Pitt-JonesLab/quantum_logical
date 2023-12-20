"""Subclass Layout to make an EncodedLayout object.

rather than mapping virtual to physical bits, we need to map logical
virtual to encoded virtual bits. the reason we need to subclass it is
because the virtual bits can be a list (e.g. repetition code) the
logical qubit 0 is now encoded qubits [0,1,2]. we might find it useful
to have logical qubits point to both a quantum register and a classical
register (where the syndrome measurements take place).
"""
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitError, EquivalenceLibrary, Qubit
from qiskit.circuit.equivalence import (
    EdgeData,
    Equivalence,
    Key,
    NodeData,
    _raise_if_param_mismatch,
)
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import LayoutError, TranspilerError
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

    @classmethod
    def from_num_qubits_ancilla(cls, num_qubits, num_ancilla):
        """Instantiate EncodedRegisters given number of qubits/ancilla."""
        if num_qubits is None or num_ancilla is None:
            raise ValueError("num_qubits and num_ancilla must be specified.")

        codeword_register = QuantumRegister(num_qubits)
        ancilla_register = QuantumRegister(num_ancilla)
        classical_codeword_register = ClassicalRegister(num_qubits)
        classical_ancilla_register = ClassicalRegister(num_ancilla)

        return cls(
            codeword_register,
            ancilla_register,
            classical_codeword_register,
            classical_ancilla_register,
        )

    def __iter__(self):
        """Yield each register for unpacking."""
        yield self.codeword_register
        yield self.ancilla_register
        yield self.classical_codeword_register
        yield self.classical_ancilla_register

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


class EncodedEquivalenceLibrary(EquivalenceLibrary):
    """Subclass of EquivalenceLibrary with a modified shape mismatch check.

    This class is specifically designed for scenarios where gates are mapped to
    encoded circuits with a different number of qubits. The standard shape mismatch
    check in EquivalenceLibrary is overridden to accommodate the encoding process,
    where the number of qubits in the equivalent circuit is a multiple of the number
    of qubits in the gate, based on the code_length.

    The `set_entry` and `add_equivalence` methods are overridden to replace the
    standard shape mismatch check with the custom one defined in this class.
    """

    def __init__(self, code_length):
        """Initialize the library with a specific code length.

        Args:
            code_length (int): The code length to be used for shape mismatch checks.
        """
        super().__init__()
        self.code_length = code_length
        # TODO: Considerations for clbits can be added here if necessary.

    def custom_shape_mismatch_function(self, gate, circuit):
        """Check the shape mismatch between gate and circuit.

        Custom function to check the shape mismatch between a gate and a
        circuit based on the encoding scheme.

        Args:
            gate: The gate to be checked.
            circuit: The circuit to be checked.

        Raises:
            CircuitError: If the number of qubits in the gate times the code_length
                          does not equal the number of qubits in the circuit, or
                          if the number of clbits does not match.
        """
        if (
            self.code_length * gate.num_qubits != circuit.num_qubits
            or gate.num_clbits != circuit.num_clbits
        ):
            raise CircuitError(
                "Cannot add equivalence between circuit and gate "
                "of different shapes. Gate: {} qubits and {} clbits. "
                "Circuit: {} qubits and {} clbits.".format(
                    gate.num_qubits,
                    gate.num_clbits,
                    circuit.num_qubits,
                    circuit.num_clbits,
                )
            )

    def set_entry(self, gate, entry):
        """Set the equivalence record for a Gate.

        Future queries for the Gate will return only the circuits provided.

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            entry (List['QuantumCircuit']) : A list of QuantumCircuits, each
                equivalently implementing the given Gate.
        """
        for equiv in entry:
            # _raise_if_shape_mismatch(gate, equiv)
            self.custom_shape_mismatch_function(gate, equiv)
            _raise_if_param_mismatch(gate.params, equiv.parameters)

        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equivs = [
            Equivalence(params=gate.params.copy(), circuit=equiv.copy())
            for equiv in entry
        ]

        self._graph[self._set_default_node(key)] = NodeData(key=key, equivs=equivs)

    def add_equivalence(self, gate, equivalent_circuit):
        """Add a new equivalence to the library.

        Future queries for the Gate will include the given circuit, in addition to all
        existing equivalences (including those from base).

        Parameterized Gates (those including `qiskit.circuit.Parameters` in their
        `Gate.params`) can be marked equivalent to parameterized circuits,
        provided the parameters match.

        Args:
            gate (Gate): A Gate instance.
            equivalent_circuit (QuantumCircuit): A circuit equivalently
                implementing the given Gate.
        """
        # _raise_if_shape_mismatch(gate, equivalent_circuit)
        self.custom_shape_mismatch_function(gate, equivalent_circuit)
        _raise_if_param_mismatch(gate.params, equivalent_circuit.parameters)

        key = Key(name=gate.name, num_qubits=gate.num_qubits)
        equiv = Equivalence(
            params=gate.params.copy(), circuit=equivalent_circuit.copy()
        )

        target = self._set_default_node(key)
        self._graph[target].equivs.append(equiv)

        sources = {
            Key(name=instruction.operation.name, num_qubits=len(instruction.qubits))
            for instruction in equivalent_circuit
        }
        edges = [
            (
                self._set_default_node(source),
                target,
                EdgeData(
                    index=self._rule_count,
                    num_gates=len(sources),
                    rule=equiv,
                    source=source,
                ),
            )
            for source in sources
        ]
        self._graph.add_edges_from(edges)
        self._rule_count += 1


class LogicalEncodingTransform(TransformationPass):
    """Expansion pass for encoded circuits.

    Transformation pass to expand a logical circuit into an encoded
    circuit with interleaved stabilizer measurements.
    """

    def __init__(
        self, encoding, equiv_lib=None, target_basis=None, stabilizer_interval=100
    ):
        """Initialize the LogicalEncodingTransform.

        Args:
            encoding: The encoding scheme to be used.
            equiv_lib: Equivalence library for the encoding.
            target_basis: Target basis gates for the encoding.
            stabilizer_interval: Interval (in terms of quantum gate duration) to insert stabilizer measurements.
        """
        # TODO
        # self.equivalence_library = self.encoding.get_equivalence_library()
        # self.target_basis = self.encoding.get_supported_basis_gates()
        super().__init__()
        self.encoding = encoding
        self.equivalence_library = equiv_lib
        self.target_basis = target_basis  # physical gates
        self.stabilizer_interval = stabilizer_interval
        self.logical_encoded_layout = EncodedLayout()
        self.registers = []

    def run(self, dag):
        """Run the transformation pass on a directed acyclic graph (DAG).

        Args:
            dag: Directed acyclic graph (DAG) representation of a quantum circuit.

        Returns:
            Transformed DAG with encoded circuit.
        """
        self._initialize_registers(dag)
        temp_qc = QuantumCircuit(*self.registers)

        self._apply_encoding_circuit(dag, temp_qc)

        self._place_logical_gates_with_stabilizers(dag, temp_qc)

        self._apply_decoding_and_measurement(dag, temp_qc)

        return circuit_to_dag(temp_qc)

    def _initialize_registers(self, dag):
        """Initialize registers for each logical qubit in the DAG."""
        encoding_codeword_len = self.encoding.code_length
        encoding_ancilla_len = self.encoding.num_ancilla

        for logical_qubit in dag.qubits:
            qreg = EncodedRegisters.from_num_qubits_ancilla(
                encoding_codeword_len, encoding_ancilla_len
            )
            qr, ar, cqr, car = qreg
            self.registers.extend([qr, ar, cqr, car])
            self.logical_encoded_layout.add(logical_qubit, qr, ar, cqr, car)

    def _apply_encoding_circuit(self, dag, temp_qc):
        """Apply the encoding circuit to each logical qubit."""
        for logical_qubit in dag.qubits:
            regs = self.logical_encoded_layout.get_logical_to_encoded_mapping()[
                logical_qubit
            ]
            self.encoding.encoding_circuit(temp_qc, regs)
        temp_qc.barrier()

    def _place_logical_gates_with_stabilizers(self, dag, temp_qc):
        """Place logical gates alternating with stabilizer measurements.

        Args:
            dag: DAG representation of the quantum circuit.
            temp_qc: Temporary QuantumCircuit being constructed.

        TODO:
        - Expand the circuit and schedule it to get a more well-defined sense of durations.
        - After scheduling, revisit the circuit to insert stabilizers at appropriate intervals.
        This will ensure stabilizers are inserted based on the actual timing of the operations.
        """
        current_duration = 0
        for layer in dag.layers():
            layer_duration = 0

            for logical_op in layer["graph"].op_nodes():
                # TODO: how do we handle parametric gates?
                # see qiskit.transpiler.passes.basis.basis_translator
                try:
                    encoded_op = self.equivalence_library.get_entry(logical_op.op)[0]
                except IndexError:
                    raise TranspilerError(
                        f"Could not find equivalence for {logical_op.op}."
                    )

                encoded_qargs = [
                    self.logical_encoded_layout.get_logical_to_encoded_mapping()[
                        qarg
                    ].codeword_register
                    for qarg in logical_op.qargs
                ]
                qarg_qubits = sum([q._bits for q in encoded_qargs], [])
                temp_qc.compose(encoded_op, qubits=qarg_qubits, inplace=True)

                # Add a default duration if it's not defined
                op_duration = (
                    encoded_op.duration if encoded_op.duration is not None else 50
                )
                layer_duration = max(layer_duration, op_duration)

            current_duration += layer_duration

            # Check if the accumulated duration exceeds the interval and apply stabilizer if it does
            if current_duration >= self.stabilizer_interval:
                temp_qc.barrier()
                self._apply_stabilizer(temp_qc)
                current_duration = 0

    def _apply_stabilizer(self, temp_qc):
        """Apply stabilizer measurements to the circuit."""
        for (
            logical_qubit_regs
        ) in self.logical_encoded_layout.get_logical_to_encoded_mapping().values():
            self.encoding.extract_syndrome(temp_qc, logical_qubit_regs)
        temp_qc.barrier()

        for (
            logical_qubit_regs
        ) in self.logical_encoded_layout.get_logical_to_encoded_mapping().values():
            self.encoding.measure_syndrome(temp_qc, logical_qubit_regs)
        temp_qc.barrier()

    def _apply_decoding_and_measurement(self, dag, temp_qc):
        """Apply decoding and perform measurements on each logical qubit."""
        for logical_qubit in dag.qubits:
            regs = self.logical_encoded_layout.get_logical_to_encoded_mapping()[
                logical_qubit
            ]
            self.encoding.decoding_circuit(temp_qc, regs)
        temp_qc.barrier()
        for logical_qubit in dag.qubits:
            regs = self.logical_encoded_layout.get_logical_to_encoded_mapping()[
                logical_qubit
            ]
            temp_qc.measure(regs.codeword_register, regs.classical_codeword_register)
