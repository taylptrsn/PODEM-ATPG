import copy
import random
from collections import namedtuple
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.offline import plot


def load_circuit(filename, test_vector):
    """
    Load the circuit instructions from the input file.

    Args:
        filename (str): Name of the file containing circuit description with ".txt" included
        test_vector (str): Initial values for circuit inputs

    Returns:
        Tuple containing circuit list, output list, and IO dictionary
    """
    logic_file = open(filename)
    logic = logic_file.readlines()
    logic_file.close()

    Instruction = namedtuple('Instruction', 'gate input output')
    circuit = []
    io_dict = {}
    output_list = []

    max_input = 0

    for line in logic:
        parse_line = line.split()
        try:
            gate = parse_line.pop(0)
        except:
            # in case there is a new line at the end of the file
            gate = ""

        if gate in ["INV", "BUF"]:
            circuit.append(
                Instruction(gate=gate,
                            input=[parse_line[0]],
                            output=parse_line[1]))
            if max(int(parse_line[0]), int(parse_line[1])) > max_input:
                max_input = max(int(parse_line[0]), int(parse_line[1]))
        elif gate in ["AND", "NAND", "OR", "NOR"]:
            circuit.append(
                Instruction(gate=gate,
                            input=[parse_line[0], parse_line[1]],
                            output=parse_line[2]))
            if max(int(parse_line[0]), int(parse_line[1]),
                   int(parse_line[2])) > max_input:
                max_input = max(int(parse_line[0]), int(parse_line[1]),
                                int(parse_line[2]))
        elif gate in ["INPUT", "OUTPUT"]:
            parse_line.pop(len(parse_line) - 1)
            if gate == "INPUT":
                for i, input_line in enumerate(parse_line):
                    io_dict[input_line] = int(test_vector[i])
            else:
                output_list = parse_line
                for output_line in parse_line:
                    io_dict[output_line] = -1

    # initialize any intermediate wires
    for i in range(1, max_input + 1):
        if str(i) not in io_dict.keys():
            io_dict[str(i)] = -1

    return circuit, output_list, io_dict, max_input


def load_circuit(filename, test_vector):
    """
    Load the circuit instructions from the input file.

    Args:
        filename (str): Name of the file containing circuit description with ".txt" included
        test_vector (str): Initial values for circuit inputs

    Returns:
        Tuple containing circuit list, output list, and IO dictionary
    """
    # Convert X to 0 in test vector
    test_vector = test_vector.replace('X', '0')

    logic_file = open(filename)
    logic = logic_file.readlines()
    logic_file.close()

    Instruction = namedtuple('Instruction', 'gate input output')
    circuit = []
    io_dictionary = {}
    output_list = []

    max_input = 0

    for line in logic:
        parse_line = line.split()
        try:
            gate = parse_line.pop(0)
        except:
            # in case there is a new line at the end of the file
            gate = ""

        if gate in ["INV", "BUF"]:
            circuit.append(
                Instruction(gate=gate,
                            input=[parse_line[0]],
                            output=parse_line[1]))
            if max(int(parse_line[0]), int(parse_line[1])) > max_input:
                max_input = max(int(parse_line[0]), int(parse_line[1]))
        elif gate in ["AND", "NAND", "OR", "NOR"]:
            circuit.append(
                Instruction(gate=gate,
                            input=[parse_line[0], parse_line[1]],
                            output=parse_line[2]))
            if max(int(parse_line[0]), int(parse_line[1]),
                   int(parse_line[2])) > max_input:
                max_input = max(int(parse_line[0]), int(parse_line[1]),
                                int(parse_line[2]))
        elif gate in ["INPUT", "OUTPUT"]:
            parse_line.pop(len(parse_line) - 1)
            if gate == "INPUT":
                for i, input_line in enumerate(parse_line):
                    io_dictionary[input_line] = int(test_vector[i])
            else:
                output_list = parse_line
                for output_line in parse_line:
                    io_dictionary[output_line] = -1

    # initialize any intermediate wires
    for i in range(1, max_input + 1):
        if str(i) not in io_dictionary.keys():
            io_dictionary[str(i)] = -1

    return circuit, output_list, io_dictionary, max_input


def evaluate_circuit(circuit,
                     io_dictionary,
                     input_net_fault_flag=True,
                     net_num=None,
                     fault=None):
    """
    Evaluate the circuit given the initial IO dictionary and fault settings.

    Args:
        circuit (list): List of circuit instructions, in named tuple form.
        io_dictionary (dict): Dictionary of input, output, and intermediate wire values.
        input_net_fault_flag (bool, optional): Flag indicating if the fault is on an input value.
        net_num (int, optional): Net number where the fault is injected.
        fault (int, optional): Value of the fault to be injected.

    Returns:
        Tuple containing updated IO dictionary, initial circuit, and initial IO dictionary.
    """
    initial_circuit = copy.deepcopy(circuit)
    initial_io_dictionary = copy.deepcopy(io_dictionary)
    eval_circuits = []

    if input_net_fault_flag:
        # Fault is on an input value
        io_dictionary[net_num] = fault

    while circuit:
        for circuit_instruction in circuit:
            # Check if input(s) are available
            eval_flag = True
            for input_num in circuit_instruction.input:
                if io_dictionary[input_num] == -1:
                    eval_flag = False

            # Add to eval list if inputs are available
            if eval_flag:
                eval_circuits.append(circuit_instruction)
                circuit.remove(circuit_instruction)

        for eval_circuit in eval_circuits:
            if eval_circuit.gate == "INV":
                if io_dictionary[eval_circuit.input[0]] == 0:
                    io_dictionary[eval_circuit.output] = 1
                else:
                    io_dictionary[eval_circuit.output] = 0
            elif eval_circuit.gate == "BUF":
                io_dictionary[eval_circuit.output] = int(
                    io_dictionary[eval_circuit.input[0]])
            elif eval_circuit.gate == "AND":
                if io_dictionary[eval_circuit.input[0]] and io_dictionary[
                        eval_circuit.input[1]]:
                    io_dictionary[eval_circuit.output] = 1
                else:
                    io_dictionary[eval_circuit.output] = 0
            elif eval_circuit.gate == "NAND":
                if io_dictionary[eval_circuit.input[0]] and io_dictionary[
                        eval_circuit.input[1]]:
                    io_dictionary[eval_circuit.output] = 0
                else:
                    io_dictionary[eval_circuit.output] = 1
            elif eval_circuit.gate == "OR":
                if io_dictionary[eval_circuit.input[0]] or io_dictionary[
                        eval_circuit.input[1]]:
                    io_dictionary[eval_circuit.output] = 1
                else:
                    io_dictionary[eval_circuit.output] = 0
            elif eval_circuit.gate == "NOR":
                if io_dictionary[eval_circuit.input[0]] or io_dictionary[
                        eval_circuit.input[1]]:
                    io_dictionary[eval_circuit.output] = 0
                else:
                    io_dictionary[eval_circuit.output] = 1

            if not input_net_fault_flag:
                # If fault is on an output net, check if it's the one just calculated
                # and force the output to the fault value if it is
                if eval_circuit.output == net_num:
                    io_dictionary[eval_circuit.output] = fault

    return io_dictionary, initial_circuit, initial_io_dictionary


def format_output(io_dictionary, output_list):
    """
    Format the outputs into the expected output vector order.

    Args:
        io_dictionary (dict): Dictionary of final input, output, and intermediate wire values.
        output_list (list): List of keys for io_dictionary that are circuit outputs.

    Returns:
        str: String representing the circuit output vector (i.e., string of 1's and 0's).
    """
    output_string = []
    for i in output_list:
        output_string.append(str(io_dictionary[i]))
    return "".join(output_string)


def generate_fault_list(fault_list_filename, max_input):
    """
    Generate the list of faults to be simulated.

    Args:
        fault_list_filename (str): Name of the file containing list of faults, or None if all faults are to be simulated.
        max_input (int): Number of nets in the circuit.

    Returns:
        dict: Dictionary with net number keys and list of faults to simulate as values.
    """
    fault_sim_dict = {}
    if fault_list_filename:
        # Initialize dictionary from file
        with open(fault_list_filename) as fault_file:
            lines = fault_file.readlines()
            for line in lines:
                parse_line = line.split()
                net_num = str(parse_line[0])
                stuck_value = int(parse_line[1])
                if net_num not in fault_sim_dict.keys():
                    fault_sim_dict[net_num] = [stuck_value]
                else:
                    fault_sim_dict[net_num].append(stuck_value)
    else:
        for i in range(1, max_input + 1):
            fault_sim_dict[str(i)] = [0, 1]
    return fault_sim_dict


def simulate_fault_list(fault_sim_dict, circuit, io_dictionary, output_list):
    """
    Simulate all the faults in the fault list and save outputs to list.

    Args:
        fault_sim_dict (dict): Dictionary with net number keys and list of faults to simulate as values.
        circuit (list): List of circuit instructions, in named tuple form.
        io_dictionary (dict): Dictionary of final input, output, and intermediate wire values.
        output_list (list): List of keys for io_dictionary that are circuit outputs.

    Returns:
        list: List of circuit outputs in string form.
    """
    fault_sim_circuit_outputs = []

    for net_num, faults_to_sim in fault_sim_dict.items():
        for fault in faults_to_sim:
            input_net_fault_flag = False
            if io_dictionary[net_num] != -1:
                input_net_fault_flag = True

            fault_io_dictionary, circuit, io_dictionary = evaluate_circuit(
                circuit, io_dictionary, input_net_fault_flag, net_num, fault)
            fault_output_vector = format_output(fault_io_dictionary,
                                                output_list)
            fault_sim_circuit_outputs.append(fault_output_vector)

    return fault_sim_circuit_outputs


def save_detected_faults(filename,
                         test_vector,
                         fault_free_output_vector,
                         fault_circuit_outputs,
                         fault_sim_dict,
                         save_output=True):
    """
    Compare fault outputs to fault-free output and save detected faults to text file.

    Args:
        filename (str): Name of the circuit file being tested.
        test_vector (str): Test vector applied to the circuit.
        fault_free_output_vector (str): Expected output of the fault-free circuit.
        fault_circuit_outputs (list): Outputs of the circuits with faults.
        fault_sim_dict (dict): List of faults simulated (for ordering purposes).
        save_output (bool, optional): Flag to determine if output should be saved to file.

    Returns:
        Tuple containing a list of detected faults and the number of faults detected.
    """
    if save_output:
        outfile = open(f"{filename}_{test_vector}_detected_faults.txt", "w")

    index = 0
    num_faults_detected = 0
    fault_list = []
    for net_num, faults in fault_sim_dict.items():
        for fault in faults:
            if fault_free_output_vector != fault_circuit_outputs[index]:
                fault_list.append((int(net_num), fault))
                num_faults_detected += 1
            index += 1

    if save_output:
        # Used for general fault detection
        outfile.write(f"{filename}_{test_vector}_detected_faults.txt\n")
        outfile.write("Net# stuck-at-value\n")
        for line in sorted(fault_list):
            outfile.write("{} stuck-at-{}\n".format(line[0], line[1]))

        outfile.write("{} faults detected".format(num_faults_detected))
        outfile.close()

    return fault_list, num_faults_detected


def generate_test_vectors(num_inputs):
    """
    Randomly generate all possible binary test vectors for a circuit.

    Args:
        num_inputs (int): Number of binary inputs to the circuit (upper bound on test vectors is 2^num_inputs - 1).

    Returns:
        list: List of binary test vectors in a random order (seeded for replicability).
    """
    decimal_list = list(range(0, (2**num_inputs) - 1))
    binary_list = [bin(decimal) for decimal in decimal_list]
    test_vectors = []
    for bin_num in binary_list:
        if len(bin_num[2:]) < num_inputs:
            append = '0' * (num_inputs - len(bin_num[2:]))
            input_test_vector = append + bin_num[2:]
        else:
            input_test_vector = bin_num[2:]
        test_vectors.append(input_test_vector)
    random.Random().shuffle(test_vectors)
    return test_vectors


def save_undetected_faults(fault_sim_dict, fault_list):
    """
    Save the list of undetected faults to a text file.

    Args:
        fault_sim_dict (dict): Dictionary of all faults that were tested.
        fault_list (tuple): List of faults that were actually detected.

    Returns:
        None
    """
    with open("undetected_faults.txt", "w") as outfile:
        for net, faults in fault_sim_dict.items():
            for fault in faults:
                if (int(net), fault) not in fault_list:
                    outfile.write("{} {}\n".format(net, fault))


def run_fault_detection(test_vectors, filename):
    """
    Apply test vectors to circuit until coverage thresholds of 75% and 90% are met.

    Args:
        test_vectors (list): List of binary test vectors to apply.
        filename (str): Name of the text file containing circuit description.

    Returns:
        Tuple containing integers representing the number of test vectors needed to reach each fault coverage threshold.
    """
    faults_detected = 0
    possible_faults = 0
    fault_coverage = []
    vector_75 = -1
    vector_90 = -1
    fault_list_filename = None
    for test_vector in test_vectors:
        # Only continue trying test vectors until we hit threshold
        if vector_90 == -1:
            circuit, output_list, circuit_io_dictionary, max_input = load_circuit(
                filename, test_vector)
            fault_free_io_dictionary, initial_circuit, initial_io_dictionary = evaluate_circuit(
                circuit, circuit_io_dictionary)
            fault_free_output_vector = format_output(fault_free_io_dictionary,
                                                     output_list)
            fault_sim_dict = generate_fault_list(fault_list_filename,
                                                 max_input)

            # Initialize full fault list initially, but then only try to detect faults that haven't
            # been detected already, which get saved to this file after each test vector
            if not fault_list_filename:
                fault_list_filename = "undetected_faults.txt"
                possible_faults = float(len(fault_sim_dict.keys()) * 2)

            fault_circuit_outputs = simulate_fault_list(
                fault_sim_dict, initial_circuit, initial_io_dictionary,
                output_list)
            # Now call save_detected_faults with all required arguments
            fault_list, num_detected = save_detected_faults(
                filename, test_vector, fault_free_output_vector,
                fault_circuit_outputs, fault_sim_dict)
            # Saved undetected faults to file for next vector to try to detect
            save_undetected_faults(fault_sim_dict, fault_list)

            # Book-keeping
            faults_detected += num_detected
            fault_coverage.append(faults_detected)
            if faults_detected / possible_faults >= 0.75:
                if faults_detected / possible_faults >= 0.9:
                    vector_90 = len(fault_coverage)
                else:
                    vector_75 = len(fault_coverage)
        else:
            break

    # Plotting
    num_test = [i for i in range(1, len(fault_coverage) + 1)]
    percent_coverage = [x / possible_faults * 100 for x in fault_coverage]

    trace = go.Scatter(x=num_test,
                       y=percent_coverage,
                       mode='lines',
                       name='Fault Coverage')

    # Add horizontal lines at 75% and 90%
    shapes = [
        dict(type='line',
             x0=1,
             x1=len(num_test),
             y0=75,
             y1=75,
             line=dict(color='red', width=2, dash='dash'),
             name='75% Threshold'),
        dict(type='line',
             x0=1,
             x1=len(num_test),
             y0=90,
             y1=90,
             line=dict(color='green', width=2, dash='dash'),
             name='90% Threshold')
    ]

    layout = go.Layout(
        title="Fault Coverage as a Function of Randomly Applied Test Vectors",
        yaxis=dict(title='Fault Coverage Percent (%)'),
        xaxis=dict(title='Number of Random Test Vectors'),
        template="plotly_dark",  # Apply dark theme
        shapes=shapes)

    fig = go.Figure(data=[trace], layout=layout)
    plot(fig)

    return vector_75, vector_90
