import datetime
import time
from string import strip

# Controller libraries
import controller.LeapIO as io
from controller import LeapDataOptimizer
from controller import LeapDataCombiner as combiner

# View libraries
from view.prompt import Prompter as prompter


def show():
    # Shows menu for training gesture data
    done = False
    while done is False:
        print("-------------")
        print("DATA TRAINING")
        print("-------------")
        print("(1) - Train")
        print("(2) - Create New Sets")
        print("(0) - Back")

        choice = raw_input("Your Choice: ")

        if choice == '1':
            show_training()
            pass
        if choice == '2':
            show_set_combiner()
            pass
        elif choice == '0':
            done = True
            pass
        else:
            print("Please try again.")
            pass

def show_training():
    done = False
    while done is False:
        print("-------------")
        print("TRAINING MODE")
        print("-------------")
        print("(1) - Individual Training")
        print("(2) - Directory Training")
        print("(3) - Full Training")
        print("(0) - Back")

        choice = raw_input("Your Choice: ")

        if choice == '1':
            individual_training()
            done = True
            pass
        elif choice == '2':
            directory_training()
            done = True
            pass
        elif choice == '3':
            full_training()
            done = True
            pass
        elif choice == '0':
            done = True
            pass
        else:
            print("Please try again.")
            pass

def show_set_combiner():
    # Shows menu for creating new data sets
    done = False
    while done is False:
        print("----------------")
        print("DATA COMBINATION")
        print("----------------")
        print("(1) - Combine Gesture Set with Individual Subject")
        print("(2) - Combine Subjects with Individual Gesture Set")
        print("(3) - Combine ALL Gesture Set with All Subjects")
        print("(0) - Back")

        choice = raw_input("Your Choice: ")
        if choice == '1':
            combiner.combine_gestures_separate_subjects()
            pass
        elif choice == '2':
            combiner.combine_subjects_separate_gestures()
            pass
        elif choice == '3':
            combiner.combine_subjects_combine_gestures()
            pass
        elif choice == '0':
            done = True
            pass
        else:
            print("Please try again.")
    pass

''' TRAINING TYPES '''

def directory_training():
    # Prompt directory to train from
    directory = prompter.prompt_data_directories()
    # Obtain data files
    data_files = io.get_data_files(directory=directory)
    # Prompt for classifier types to train with
    classifier_types = prompter.prompt_classifier()

    do_multi_training(data_files=data_files, classifier_types=classifier_types)
    pass

def individual_training():
    file_name = None  # This indicates that the file name has not been set - will create a new one instead

    # Prompt for data file to train from
    data_file = prompter.prompt_data_file()

    done = False
    while done is False:
        # Prompt for classifier types to train with
        classifier_types = prompter.prompt_classifier()
        if len(classifier_types) > 1:
            print("Please try again - please choose a single classifier")
        else:
            done = False
            do_single_training(data_file=data_file, classifier_type=classifier_types[0])

def full_training():
    # Obtain all data files
    raw_data_files = io.get_data_files(combined=False)
    com_data_files = io.get_data_files(combined=True)

    # Join the two lists
    data_files = raw_data_files + com_data_files

    for data in data_files:
        print data
    # Benchmarking timer
    initial = time.time()

    # Since classifier types not specified - will default to all classifier types
    do_multi_training(data_files=data_files)

    # Benchmarking timer calculation and output
    print("* * * BENCHMARK TIMER RESULTS * * * ")
    final = time.time()
    elapsed_time = final - initial
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    print("* * * * * * * * * * * * * * * * * * ")


''' TRAINING FUNCTIONS '''

def do_multi_training(data_files, classifier_types=None):
    # Default
    if classifier_types is None:
        classifier_types = ['svm', 'nn', 'dt']

    # Train for each data file
    for data_file in data_files:
        # Train with each classifier type
        for classifier_type in classifier_types:
            do_single_training(data_file, classifier_type)

def do_single_training(data_file, classifier_type):
    file_name = None  # This indicates that the file name has not been set - will create a new one instead
    params = []

    # Obtain relevant information based on file name
    subject_name = data_file.rsplit("(", 1)[1].rsplit(")")[0]
    feature_set = strip(data_file.rsplit(")")[1].rsplit(".")[0].rsplit("--")[1])
    gesture_set = strip(data_file.split(")")[1].split("--")[0])

    print("Data File >> " + data_file)
    print("    Classifier  - " + classifier_type)
    print("    Feature Set - " + feature_set)
    print("    Gesture Set - " + gesture_set)

    if classifier_type == 'svm':
        # Obtain relevant hyperparameter
        kernel_list = io.read_col("kernels.txt")
        for kernel in kernel_list:
            # Reset params before populating
            params = []
            params.append(kernel)
        pass
    elif classifier_type == 'nn':
        # Obtain relevant hyperparameter
        activations = ['relu', 'logistic']
        for activation in activations:
            # Reset params before populating
            params = []
            params.append(activation)
        pass
    elif classifier_type == 'dt':
        # Obtain relevant hyperparameter
        criterions = ['gini', 'entropy']
        for criterion in criterions:
            # Reset params before populating
            params = []
            params.append(criterion)

        pass
    else:
        return None

    # Create Report and Summary
    training_summary = train_auto(csv_file=data_file, subject_name=subject_name,
                                       feature_type=feature_set, gesture_set=gesture_set,
                                       classifier_type=classifier_type, params=params)
    file_name = io.save_report(file_name=file_name, subject_name=subject_name, report_header='training',
                               line=training_summary, classifier_type=classifier_type,
                               gesture_set=gesture_set, feature_set=feature_set)
    pass

def train_auto(csv_file, subject_name, feature_type, gesture_set, classifier_type, params):
    results = LeapDataOptimizer.obtain_optimal_classifier(
        csv_file_name=csv_file,
        subject_name=subject_name,
        feature_type=feature_type,
        gesture_set=gesture_set,
        classifier_type=classifier_type,
        params=params
    )

    optimal_classifier = results[0]
    training_summary = results[1]

    optimal_classifier.save_classifier()

    return training_summary
