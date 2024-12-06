import copy
import random
from collections import namedtuple
from typing import List, Dict, Optional, Tuple

# Define basic circuit instruction structure
Instruction = namedtuple('Instruction', 'gate input output')


def inverse(boolean_value):
    """
    Returns the inverse of a binary value unless the value is undefined.
    Args:
        boolean_value (int): Integer representing the binary value, can be 0, 1, or -1 (undefined).
    Returns:
        int: Inverse of the given boolean value. Returns 0 if input is 1, returns 1 if input is 0, and returns -1 if value is undefined.
    """
    if boolean_value == 1:
        return 0
    elif boolean_value == 0:
        return 1
    else:
        return -1


def and_nand(var1, var2, not_flag=False):
    """
    Evaluates AND or NAND operation for binary values with possible defect values.
    Args:
        var1 (int): First circuit input value, can be 0, 1, D (2), D' (-2), or x (-1).
        var2 (int): Second circuit input value, can be 0, 1, D (2), D' (-2), or x (-1).
        not_flag (bool): Indicator for NAND operation if True, AND operation if False.
    Returns:
        int: Result of the AND/NAND operation for the given values.
    """
    # normal case
    if var1 in [0, 1] and var2 in [0, 1]:
        if not_flag:
            return inverse(var1 & var2)
        else:
            return var1 & var2
    # no defect propagated
    elif var1 == 0 or var2 == 0:
        if not_flag:
            return 1
        else:
            return 0
    # if either input is unknown then we don't know if defect will propagate
    elif var1 == -1 or var2 == -1:
        return -1
    elif var1 == -2 or var2 == -2:
        if var1 == -2 and var2 != 2 or var1 != 2 and var2 == -2:
            if not_flag:
                return 2
            else:
                return -2
        else:
            if not_flag:
                return 1
            else:
                return 0
    elif var1 == 2 or var2 == 2:
        if var1 == 2 and var2 != -2 or var1 != -2 and var2 == 2:
            if not_flag:
                return -2
            else:
                return 2
        else:
            if not_flag:
                return 1
            else:
                return 0


def or_nor(var1, var2, not_flag=False):
    """
    Evaluates OR or NOR operation for binary values with possible defect values.
    Args:
        var1 (int): First circuit input value, can be 0, 1, D (2), D' (-2), or x (-1).
        var2 (int): Second circuit input value, can be 0, 1, D (2), D' (-2), or x (-1).
        not_flag (bool): Indicator for NOR operation if True, OR operation if False.
    Returns:
        int: Result of the OR/NOR operation for the given values.
    """
    # normal case
    if var1 in [0, 1] and var2 in [0, 1]:
        if not_flag:
            return inverse(var1 | var2)
        else:
            return var1 | var2
    # no defect propagated
    elif var1 == 1 or var2 == 1:
        if not_flag:
            return 0
        else:
            return 1
    elif var1 == -1 or var2 == -1:
        return -1
    elif var1 == -2 or var2 == -2:
        if var1 == -2 and var2 != 2 or var1 != 2 and var2 == -2:
            if not_flag:
                return 2
            else:
                return -2
        else:
            if not_flag:
                return 0
            else:
                return 1
    elif var1 == 2 or var2 == 2:
        if var1 == 2 and var2 != -2 or var1 != -2 and var2 == 2:
            if not_flag:
                return -2
            else:
                return 2
        else:
            if not_flag:
                return 0
            else:
                return 1


class PodemSim:
    """
    PODEM (Path-Oriented Decision Making) Algorithm Implementation
    This class implements the PODEM algorithm for automatic test pattern generation (ATPG)
    to detect stuck-at faults in digital circuits.
    Special Values Used:
        -1 (UNDEFINED): Represents an undefined logic value
        -2 (STUCK_AT_0): Represents a stuck-at-0 fault (D')
         2 (STUCK_AT_1): Represents a stuck-at-1 fault (D)
         0: Logic low
         1: Logic high
    """
    UNDEFINED = -1  # Represents an undefined logic value, Undefined/Dont Care
    STUCK_AT_0 = -2  # Represents a stuck-at-0 fault (D')
    STUCK_AT_1 = 2  # Represents a stuck-at-1 fault (D)

    def __init__(self, net_num: int, stuck_at_value: int):
        """
        Initialize PODEM algorithm with fault parameters.
        Args:
            fault_net (int): The net number where the fault is located
            stuck_value (int): The fault type (0 = stuck-at-0, 1 = stuck-at-1)
        """
        self.net_num = str(net_num)
        self.stuck_at_value = stuck_at_value
        self.circuit: List[Instruction] = []
        self.io_dictionary: Dict[str, int] = {}
        self.previous_io_dicts: List[Dict[str, int]] = []
        self.input_list: List[str] = []
        self.output_list: List[str] = []
        self.dfrontier: List[Instruction] = []
        self.initialized = False

    def inject_fault(self) -> None:
        """Injects the specified stuck-at fault into the circuit."""
        self.io_dictionary[
            self.
            net_num] = self.STUCK_AT_1 if self.stuck_at_value else self.STUCK_AT_0

    def error_at_primary_output(self) -> bool:
        """
        Checks if the fault effect has propagated to a primary output. (D/D')
        Returns:
            bool: True if the fault effect is present at the output, else False
        """
        return any(
            abs(self.io_dictionary[output_net]) == 2
            for output_net in self.output_list)

    def update_dfrontier(self) -> None:
        """
        Updates and returns the current D-frontier gates.
        Returns:
            List[Instruction]: A list of gates on the D-frontier
        """
        dfrontier = []
        for circuit_instruction in self.circuit:
            if len(circuit_instruction.input) > 1:
                fault_flag = any(
                    abs(self.io_dictionary[input_num]) == 2
                    for input_num in circuit_instruction.input)
                undefined_flag = any(
                    self.io_dictionary[input_num] == self.UNDEFINED
                    for input_num in circuit_instruction.input)
                if fault_flag and undefined_flag:
                    dfrontier.append(circuit_instruction)
        self.dfrontier = dfrontier

    def load_circuit(self,
                     filename: str,
                     test_vector: Optional[str] = None) -> None:
        """
        Loads circuit description from file and initializes internal structures.
        Args:
            filename (str): The file name containing circuit description
            test_vector (Optional[str]): Initial logic values for circuit inputs
        """
        with open(filename) as logic_file:
            logic = logic_file.readlines()
        circuit = []
        io_dictionary = {}
        max_input = 0
        for line in logic:
            parse_line = line.split()
            if not parse_line:
                continue
            gate = parse_line.pop(0)
            if gate in ["INV", "BUF"]:
                circuit.append(
                    Instruction(gate=gate,
                                input=[parse_line[0]],
                                output=parse_line[1]))
                max_input = max(max_input, int(parse_line[0]),
                                int(parse_line[1]))
            elif gate in ["AND", "NAND", "OR", "NOR"]:
                circuit.append(
                    Instruction(gate=gate,
                                input=[parse_line[0], parse_line[1]],
                                output=parse_line[2]))
                max_input = max(max_input, int(parse_line[0]),
                                int(parse_line[1]), int(parse_line[2]))
            elif gate in ["INPUT", "OUTPUT"]:
                parse_line.pop(len(parse_line) - 1)
                if gate == "INPUT":
                    if test_vector:
                        for i, input_line in enumerate(parse_line):
                            io_dictionary[input_line] = int(test_vector[i])
                    else:
                        self.input_list = parse_line
                        for input_line in parse_line:
                            io_dictionary[input_line] = self.UNDEFINED
                else:
                    self.output_list = parse_line
                    for output_line in parse_line:
                        io_dictionary[output_line] = self.UNDEFINED
        # Initialize intermediate wires
        for i in range(1, max_input + 1):
            if str(i) not in io_dictionary:
                io_dictionary[str(i)] = self.UNDEFINED
        self.circuit = circuit
        self.io_dictionary = io_dictionary

    def evaluate_circuit(self) -> None:
        """Evaluate the circuit with current inputs"""
        circuit = copy.deepcopy(self.circuit)
        eval_circuits = []
        num_evals_prev = float('inf')
        while circuit:
            eval_circuits = [
                instr for instr in circuit if self.io_dictionary[instr.output]
                in [self.UNDEFINED, self.STUCK_AT_0, self.STUCK_AT_1]
            ]
            num_evals = len(eval_circuits)
            while eval_circuits:
                eval_circuit = eval_circuits.pop(0)
                self._evaluate_gate(eval_circuit)
                if eval_circuit.output == self.net_num:
                    if self.io_dictionary[eval_circuit.output] == inverse(
                            self.stuck_at_value):
                        self.inject_fault()
                        self.initialized = True
            if num_evals == num_evals_prev:
                break
            num_evals_prev = num_evals
        self.update_dfrontier()

    def _evaluate_gate(self, eval_circuit: Instruction) -> None:
        """Helper method to evaluate a single gate"""
        if eval_circuit.gate == "INV":
            val = self.io_dictionary[eval_circuit.input[0]]
            if val == 0:
                self.io_dictionary[eval_circuit.output] = 1
            elif val == 1:
                self.io_dictionary[eval_circuit.output] = 0
            elif val == self.STUCK_AT_1:
                self.io_dictionary[eval_circuit.output] = self.STUCK_AT_0
            elif val == self.STUCK_AT_0:
                self.io_dictionary[eval_circuit.output] = self.STUCK_AT_1
        elif eval_circuit.gate == "BUF":
            self.io_dictionary[eval_circuit.output] = self.io_dictionary[
                eval_circuit.input[0]]
        elif eval_circuit.gate == "AND":
            self.io_dictionary[eval_circuit.output] = and_nand(
                self.io_dictionary[eval_circuit.input[0]],
                self.io_dictionary[eval_circuit.input[1]])
        elif eval_circuit.gate == "NAND":
            self.io_dictionary[eval_circuit.output] = and_nand(
                self.io_dictionary[eval_circuit.input[0]],
                self.io_dictionary[eval_circuit.input[1]], True)
        elif eval_circuit.gate == "OR":
            self.io_dictionary[eval_circuit.output] = or_nor(
                self.io_dictionary[eval_circuit.input[0]],
                self.io_dictionary[eval_circuit.input[1]])
        elif eval_circuit.gate == "NOR":
            self.io_dictionary[eval_circuit.output] = or_nor(
                self.io_dictionary[eval_circuit.input[0]],
                self.io_dictionary[eval_circuit.input[1]], True)

    def objective(self) -> Tuple[str, int]:
        """
        Determines the next objective for the PODEM algorithm.
        Returns:
            Tuple[str, int]: The net and required logic value
        """
        if self.net_num in self.input_list and self.io_dictionary[
                self.net_num] == self.UNDEFINED:
            self.initialized = True
            return self.net_num, self.STUCK_AT_1 if self.stuck_at_value else self.STUCK_AT_0
        elif self.io_dictionary[self.net_num] == self.UNDEFINED:
            self.initialized = False
            return self.net_num, inverse(self.stuck_at_value)
        else:
            circuit_instruction = self.dfrontier.pop(0)
            controlling_val = 0 if "AND" in circuit_instruction.gate else 1
            for input_num in circuit_instruction.input:
                if self.io_dictionary[input_num] == self.UNDEFINED:
                    return input_num, inverse(controlling_val)
        return "", 0

    def backtrack(self, net_num: str, value: int) -> Tuple[str, int]:
        """
        Maps a target objective to a primary input assignment.
        Args:
            target_net (str): The target net
            target_value (int): The desired value for the target
        Returns:
            Tuple[str, int]: The net and value for primary input
        """
        while net_num not in self.input_list:
            for circuit_instruction in self.circuit:
                if circuit_instruction.output == net_num:
                    if circuit_instruction.gate == "INV":
                        net_num = circuit_instruction.input[0]
                        value = inverse(value)
                    elif circuit_instruction.gate in ["NAND", "NOR"]:
                        net_num = circuit_instruction.input[
                            0] if self.io_dictionary[circuit_instruction.input[
                                0]] == self.UNDEFINED else circuit_instruction.input[
                                    1]
                        value = value ^ 1
                    elif circuit_instruction.gate in ["AND", "OR"]:
                        net_num = circuit_instruction.input[
                            0] if self.io_dictionary[circuit_instruction.input[
                                0]] == self.UNDEFINED else circuit_instruction.input[
                                    1]
                        value = value ^ 0
                    else:  # buffer
                        net_num = circuit_instruction.input[0]
                    break
        return net_num, value

    def imply(self, net_num: str, value: int) -> None:
        """
        Assigns a value to a net and evaluates its affect on the circuit.
        Args:
            target_net (str): The target net number
            value (int): The logic value to assign
        """
        self.previous_io_dicts.append(copy.deepcopy(self.io_dictionary))
        self.io_dictionary[net_num] = value
        self.evaluate_circuit()
        self.update_dfrontier()

    def reverse(self) -> None:
        """Reverts to a previously saved circuit state."""
        if self.previous_io_dicts:
            self.io_dictionary = self.previous_io_dicts.pop()

    def podem(self) -> bool:
        """
        Executes the PODEM algorithm to find a test vector.
        Returns:
            bool: True if a test vector is found, else False
        """
        # Check termination conditions
        if self.error_at_primary_output():
            return True
        if self.io_dictionary[self.net_num] == self.stuck_at_value:
            return False
        if self.initialized and not self.dfrontier:
            return False
        # Get next objective and backtrack to primary input
        target_net, target_value = self.objective()
        pi_net, pi_value = self.backtrack(target_net, target_value)
        # Try first assignment
        self.imply(pi_net, pi_value)
        if self.podem():
            return True
        # Try complementary assignment
        self.reverse()
        complement_value = inverse(pi_value) if abs(
            pi_value) != 2 else pi_value
        self.imply(pi_net, complement_value)
        if self.podem():
            return True
        # Backtrack if both assignments fail
        self.reverse()
        self.imply(pi_net, self.UNDEFINED)
        return False

    def format_output(self, output_list: List[str]) -> str:
        """
        Formats the circuit outputs into a string representation.
        Args:
            output_nets (List[str]): List of circuit output nets
        Returns:
            str: Concatenated string of output values
        """
        output_string = []
        for i in output_list:
            if self.io_dictionary[i] == self.UNDEFINED:
                output_string.append('X')
            elif self.io_dictionary[i] == self.STUCK_AT_0:
                output_string.append('1')
            elif self.io_dictionary[i] == self.STUCK_AT_1:
                output_string.append('0')
            else:
                output_string.append(str(self.io_dictionary[i]))
        return "".join(output_string)
