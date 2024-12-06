from podem import *
# run_podem.py
""" SCRIPT USER INPUTS START """
#relative to script path
#FILENAME = "s27.txt"
#NET_NUM = 179
#STUCK_AT_VALUE = 0  # 0 or 1

#FILENAME = "s298f_2.txt"
#NET_NUM = 179
#STUCK_AT_VALUE = 0  #0 or 1

#FILENAME = "s344f_2.txt"
#NET_NUM = 179
#STUCK_AT_VALUE = 0  #0 or 1

FILENAME = "s349f_2.txt"
NET_NUM = 179
STUCK_AT_VALUE = 1  #0 or 1
""" SCRIPT USER INPUTS END """

# User inputs


def main():
    print(FILENAME)
    podem_runner = PodemSim(NET_NUM, STUCK_AT_VALUE)
    podem_runner.load_circuit(FILENAME)
    test_possible = podem_runner.podem()
    if test_possible:
        test_vector = podem_runner.format_output(podem_runner.input_list)
        print(test_vector)
    else:
        print("Undetectable, No vector found")


if __name__ == "__main__":
    main()
