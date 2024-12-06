from deductive import *
# run_deductive.py
""" SCRIPT USER INPUTS BEGIN """
# assumes file is in the same directory as script execution
RUN_MODE = "deductive"  # coverage (for random fault coverage test) or deductive (for deductive fault detection with input test vector(s))
FILENAME = "s27.txt"
NUM_INPUTS = 7  # Number of inputs for the circuit, used for random test vector generation in coverage sim
TEST_VECTORS = ['0000000', '1001X10', '1101101']

# FILENAME = "s298f_2.txt"
#NUM_INPUTS = 17 # Number of inputs for the circuit, used for random test vector generation in coverage sim
# TEST_VECTORS = ['X1XX1XXXXXXXX00XX']

#FILENAME = "s344f_2.txt"
# NUM_INPUTS = 24 # Number of inputs for the circuit, used for random test vector generation in coverage sim
#TEST_VECTORS = ['10XXXXXXXXXXXXXXXXXXXXXX']

# FILENAME = "s349f_2.txt"
# NUM_INPUTS = 24 # Number of inputs for the circuit, used for random test vector generation in coverage sim
# TEST_VECTORS = ['101XXXXXXXXXXXX0XXXXXXXX']
""" SCRIPT USER INPUTS END """

FAULT_LIST_FILENAME = None


def process_test_vector(test_vector, filename):
    print(FILENAME)
    # Remove the for loop here - process single test vector
    circuit, output_list, circuit_io_dict, max_input = load_circuit(
        FILENAME, test_vector)
    fault_free_io_dict, initial_circuit, initial_io_dict = evaluate_circuit(
        circuit, circuit_io_dict)
    fault_free_output_vector = format_output(fault_free_io_dict, output_list)
    fault_sim_dict = generate_fault_list(FAULT_LIST_FILENAME, max_input)
    fault_circuit_outputs = simulate_fault_list(fault_sim_dict,
                                                initial_circuit,
                                                initial_io_dict, output_list)
    fault_list, fault_count = save_detected_faults(FILENAME, test_vector,
                                                   fault_free_output_vector,
                                                   fault_circuit_outputs,
                                                   fault_sim_dict)
    print(f"Detected faults: {fault_list}")
    print(f"Total faults detected: {fault_count}")


def process_fault_coverage(filename, num_inputs):
    """Process fault coverage using random test vectors"""
    print(f"Running fault coverage analysis for {filename}")
    test_vectors = generate_test_vectors(num_inputs)
    vector_75, vector_90 = run_fault_detection(test_vectors, filename)
    print(f"Vectors needed for 75% coverage: {vector_75}")
    print(f"Vectors needed for 90% coverage: {vector_90}")


def main():
    print(f"Processing circuit: {FILENAME}")
    if RUN_MODE == "deductive":
        for test_vector in TEST_VECTORS:
            process_test_vector(test_vector, FILENAME)
    elif RUN_MODE == "coverage":
        process_fault_coverage(FILENAME, NUM_INPUTS)
    else:
        print("Invalid RUN_MODE. Use 'coverage' or 'single_vector'")


if __name__ == "__main__":
    main()
