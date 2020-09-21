import controller.LeapIO as io
import controller.Printer as printer

from string import lower, upper


def prompt_subject_name():
    done = False
    while done is False:
        print("~ ~ ~ ~ ~ ~ ~ ~ SUBJECT NAMES ~ ~ ~ ~ ~ ~ ~ ~")
        subjects = io.read_col('subjects.txt')
        printer.print_numbered_list(subjects)
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")

        choice = raw_input("Enter the subject name you wish to use: ")

        if int(choice) > len(subjects) or int(choice) < 1:
            print("Please try again.")
        else:
            subject_name = subjects[int(choice) - 1]
            print("Chosen Name : " + subject_name)
            print("")
            return subject_name


def prompt_gesture_name(gesture_src='gestures_counting.txt'):
    # Prompts user for name of gesture
    done = False

    while done is False:
        gesture_list = io.read_col(gesture_src)
        print("* List of Valid Gestures *")
        printer.print_numbered_list(gesture_list)
        choice = raw_input("Enter the Gesture Name: ")
        gesture_name = lower(gesture_list[int(choice) - 1])
        print("")

        num_choice = int(choice)

        if 1 > num_choice > len(gesture_list):
            print("Please try again")
        else:
            return gesture_name


def prompt_gesture_set():
    done = False

    while done is False:
        gesture_src_list = ['gestures_counting.txt', 'gestures_status.txt', 'gestures_asl.txt']
        gesture_set_list = ['COUNTING GESTURES', 'STATUS GESTURES', 'ASL GESTURES']

        print("* List of Valid Gesture Sources *")
        printer.print_numbered_list(gesture_set_list)
        choice = raw_input("Enter the Gesture Source: ")
        print("")

        num_choice = int(choice)
        if 1 <= num_choice <= len(gesture_src_list):
            gesture_src = lower(gesture_src_list[int(choice) - 1])
            gesture_list = io.read_col(gesture_src)

            return gesture_set_list[num_choice - 1], gesture_list, gesture_src
        else:
            print("Please try again")


def prompt_iterations():
    # Prompts user for number of iterations
    done = False

    while done is False:
        iterations = raw_input("Training Size: ")

        if iterations.isdigit() is not False or iterations is not None or iterations is not "":
            done = True
            return int(iterations)
        else:
            print("Please try again")


def prompt_data_file():
    # Prompt for ALL data files
    data_files = io.get_data_files(combined=False)  # Raw
    data_files.extend(io.get_data_files(combined=True))  # Combined

    print("")
    print("~ ~ ~ ~ ~ ~ ~ ~ LIST OF DATA SOURCE ~ ~ ~ ~ ~ ~ ~ ~")
    printer.print_numbered_list(data_files)
    print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~")

    choice = raw_input("Enter the data set to train from: ")
    data_file = data_files[int(choice) - 1]
    print("Chosen File : " + data_file)
    print("")

    return data_file

def prompt_data_directories():
    done = False
    directories = [io.dat_dir, io.com_dir]
    while done is False:
        printer.print_numbered_list(directories)

        choice = raw_input("Your Choice: ")

        if choice == '1':
            return directories[0]
        elif choice == '2':
            return directories[1]
        else:
            print("Please try again - invalid input")
    pass


def prompt_classifier():
    done = False
    classifier_types = ["Support Vector Machine", "Multilayer Perceptron (Neural Network)", "Decision Trees", "All"]
    while done is False:
        printer.print_numbered_list(classifier_types)

        choice = raw_input("Your Choice: ")

        if choice == '1':
            return ['svm']
            pass
        elif choice == '2':
            return ['nn']
            pass
        elif choice == '3':
            return ['dt']
            pass
        elif choice == '4':
            return ['svm', 'nn', 'dt']
        else:
            print("Please try again - invalid choice")
