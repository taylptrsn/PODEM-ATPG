# PODEM Project ECE 6140

## Language and Dependencies
- **Language:** Python 3.11 or Above  
- **Dependencies:** Plotly 5.24.1 or Above  
- *Note:* You may also need a Python virtual environment (venv), depending on your local environment.

## Project Structure
├── deductive.py # Deductive fault simulator

├── run_deductive.py # Deductive simulator runner

├── podem.py # PODEM implementation

├── run_podem.py # PODEM runner

└── circuits/ # Circuit description files (.txt) (assumed to be in script directory)


### File Descriptions

#### 1. `deductive.py`
- Defines all the functions necessary for the deductive fault simulator.
- This file contains the code for **Project Part 1 and 2**.

#### 2. `run_deductive.py`
- Runner script that passes user input data to the functions in `deductive.py`.
- Can run either deductive fault simulation or fault coverage simulation, as defined in Project Part 2.

**Input Parameters:**
- **FILENAME:** Name of the circuit file (String).  
- **TEST_VECTOR:** Test vector to simulate through the circuit of interest (Vector of Binary Values).  
- **RUN_MODE:** Toggle to run deductive fault simulation or fault coverage simulation. Use `"deductive"` for deductive fault simulation or `"coverage"` for fault coverage.  
- **NUM_INPUTS:** Number of inputs for the circuit, used for random test vector generation.

**Output:**
- Prints a list of detected faults and the total number of detected faults.  
- Writes an output file with the detected faults and their total count, named as `<FILENAME>_<TEST_VECTOR>_detected_faults.txt`.

#### 3. `podem.py`
- Defines all the functions necessary for implementing the PODEM algorithm.
- Built on top of the deductive simulation framework.

#### 4. `run_podem.py`
- Runner script that passes user input data to the functions in `podem.py`.

**Input Parameters:**
- **FILENAME:** Name of the circuit file (String).  
- **NET_NUM:** Number of the net of interest to generate a pattern for (if a pattern exists) (Integer).  
- **STUCK_AT_VALUE:** Fault of interest to generate a pattern for (if a pattern exists) (Boolean: `1` or `0`).

**Output:**
- Prints the test vector for the net and fault of interest if detectable.  
- If undetectable, prints "Undetectable, No vector found".

---

## Running the Scripts

### For Deductive Fault Simulation:
`python run_deductive.py`

or

`python3 run_deductive.py  # Depending on environment`

For PODEM Test Pattern Generation:

`python run_podem.py`

or

`python3 run_podem.py  # Depending on environment`
