README: 
PODEM Project ECE 6140
Language: Python 3.11 or Above
Dependencies: Plotly 5.24.1 or Above
#You may also need a python virtual environment (venv), depending on your local environment

├── deductive.py            # Deductive fault simulator
├── run_deductive.py    # Deductive simulator runner
├── podem.py                 # PODEM implementation
├── run_podem.py         # PODEM runner
└── circuit_files/             # Circuit description files(.txt)(assumed to be in script directory)

1. deductive.py
-defines all the functions necessary for the deductive fault simulator, this is effectively the code for Project Part 1 and 2.
2. run_deductive.py
-Runner script that passes user input data to the functions in deductive.py
-Can run either deductive fault simulation or fault coverage simulation, as defined in project 2
INPUT:
-FILENAME: name of circuit file, String
-TEST_VECTOR: Test vector to simulate through circuit of Interest, Vector of Binary Values
-RUN_MODE: toggle to run deductive fault sim or fault coverage simulation, string “deductive” for deductive fault, “coverage” for fault coverage
-NUM_INPUTS: Number of inputs for circuit, used for random test vector generation 
OUTPUT: 
-Prints List of Detected Faults, as well as Total Number of Detected Faults
-Writes an output file with the List of Detected Faults and Total Number of Detected faults with the name <FILENAME>_<TEST_VECTOR>_detected_faults.txt	
3.podem.py
-defines all the functions necessary for the PODEM algorithm implementation, built on top of the deductive simulation framework
4.run_podem.py
-Runner script that passes user input data to the functions in deductive.py

INPUT: 
-FILENAME: name of circuit file, String
-NET_NUM: number of net of interest to generate a pattern for (if pattern exists), Integer
-STUCK_AT_VALUE: fault of interest to generate a pattern for (if pattern exists), Boolean 1 or 0
OUTPUT: 
-Prints Test Vector for Net and Fault of Interest (If detectable) or "Undetectable, No vector found" Message (if undetectable)

To run the scripts:
# For deductive fault simulation
python run_deductive.py 
or  
python3 run_deductive.py  #depending on environment

# For PODEM test pattern generation
python run_podem.py 
or  
python3 run_podem.py #depending on environment
