#!/usr/bin/env python3
""" QA Bot """


def answer_loop():
    """ A loop that wait the typing of the user """
    while True:
        response = input("Q: ")
        if response.lower() in EXIT_COMMANDS:
            print("A: Goodbye")
            break

        print("A:")


if __name__ == '__main__':
    try:
        EXIT_COMMANDS = ['exit', 'quit', 'goodbye', 'bye']
        answer_loop()
        # prevent when prec ctrl + c a keyboard error
    except KeyboardInterrupt:
        print("\nA: Goodbye")
        exit(0)
