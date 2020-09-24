import inspect
import os
import sys
from datetime import date, datetime
from string import lower, rsplit, upper
import pickle
import pandas as pd
import numpy as np

src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
lib_dir = os.path.abspath(os.path.join(src_dir, '../leapLib'))
sys.path.insert(0, lib_dir)

con_dir = os.path.dirname(os.getcwd()) + "\config\\"
out_dir = os.path.dirname(os.getcwd()) + "\output\\"

dta_dir = out_dir + "data\\"
tdd_dir = out_dir + "trained\\"

trd_dir = tdd_dir + "classifiers\\"
sca_dir = tdd_dir + "scales\\"

dat_dir = dta_dir + "training\\raw\\"
com_dir = dta_dir + "training\\combined\\"
uns_dir = dta_dir + "testing\\raw\\"
cun_dir = dta_dir + "testing\\combined\\"

sum_dir = out_dir + "reports\\"
tra_dir = sum_dir + "training\\"
cla_dir = sum_dir + "classification\\"


# DATA FILE FUNCTIONS
def save_data(file_name, subject_name, gesture_name, data_set):
    file_name = dat_dir + "(" + subject_name + ") " + file_name + ".csv"
    label_list = []
    value_list = []

    # Get lists of data and values
    for data in data_set:
        label_list.append(data.label)
        value_list.append(data.value)

    # Add class to the end of the values
    label_list.append("class")

    # Validate if file exist or if parameters match
    validate_data_file(file_name=file_name, labels=label_list)

    # After creating new file (or not), append to existing file
    append_to_data_file(gesture_name=gesture_name, file_name=file_name, data_set=value_list)


def create_data_file(file_name, labels, verbose=False):
    if verbose is True:
        print("System       :       Creating Data File " + file_name)

    writer = open(file_name, 'w')

    writer.write(",".join(labels))
    writer.write('\n')
    writer.close()


def append_to_data_file(gesture_name, file_name, data_set, verbose=False):
    if gesture_name is not None and file_name is not None:
        if verbose is True:
            print("System       :       Appending Data File " + file_name)
        values = []

        # Change list to string data
        for data in data_set:
            values.append(str(data))

        # Append the name of gesture for classification
        values.append(gesture_name)
        writer = open(file_name, "a")
        line = ",".join(values)
        writer.write(line)
        writer.write('\n')
        writer.close()


def save_classifier(pickle_name, data, verbose=False):
    file_path = trd_dir + pickle_name

    if verbose is True:
        print("Saving Classifier : " + str(file_path))

    pickle_file = open(file_path, 'wb')
    pickle.dump(data, pickle_file)
    pickle_file.close()
    pass


def load_classifier(pickle_name, verbose=False):
    file_path = trd_dir + rsplit(pickle_name, "\\")[-1]

    if verbose is True:
        print("Saving Classifier : " + str(file_path))

    pickle_file = open(file_path, 'rb')
    data = pickle.load(pickle_file)
    return data


def save_scale(pickle_name, data, verbose=False):
    file_path = sca_dir + pickle_name

    if verbose is True:
        print("Saving Scale      : " + str(file_path))

    pickle_file = open(file_path, 'wb')
    pickle.dump(data, pickle_file)
    pickle_file.close()


def load_scale(pickle_name, verbose=False):
    file_path = sca_dir + rsplit(pickle_name, "\\")[-1]

    if verbose is True:
        print("Loading Scale     : " + str(file_path))

    pickle_file = open(file_path, 'rb')
    data = pickle.load(pickle_file)
    return data


# SUMMARY FUNCTIONS
def save_report(subject_name, report_header, line, classifier_type=None, feature_set=None, gesture_set=None,
                file_name=None):
    # Validate the file name - creates new one if does not exist
    file_name = validate_report_file(file_name=file_name, report_header=report_header, subject_name=subject_name,
                                     classifier_type=classifier_type, feature_set=feature_set)
    # Append to report once validated
    append_to_report(file_name=file_name, line=line)

    return file_name


def append_to_report(file_name, line):
    if line is not None:
        writer = open(file_name, 'a')
        writer.write(line)
        writer.write('\n')
        writer.close()


# TRAINING SUMMARY FUNCTIONS
def create_training_report(subject_name, feature_set, classifier_type):
    file_name = upper(classifier_type) + "_TRAINING_REPORT "
    file_name = tra_dir + file_name + "(" + subject_name + ") " + feature_set + ".txt"

    writer = open(file_name, 'w')
    writer.close()

    return file_name


# CLASSIFICATION SUMMARY FUNCTIONS
def create_classification_report(subject_name, classifier_type=None, feature_set=None):
    file_name = upper(classifier_type) + "_CLASSIFICATION_REPORT"
    file_name = cla_dir + file_name + " (" + subject_name + ") " + feature_set + ".txt"

    writer = open(file_name, 'w')
    writer.close()

    return file_name


# VALIDATION FUNCTIONS
def validate_data_file(file_name, labels, verbose=False):
    invalid = False

    if does_file_exist(file_name):
        if not do_parameters_match(file_name=file_name, labels=labels):
            if verbose is True:
                print("System       :       Parameters do not match - creating a new file")
            invalid = True
    else:
        if verbose is True:
            print("System       :       Specified file name does not exist - creating a new file")
        invalid = True

    if invalid is True:  # Only create file if specified file is invalid
        create_data_file(file_name=file_name, labels=labels)


def validate_report_file(report_header, subject_name, file_name, classifier_type=None, feature_set=None):
    if file_name is None:
        if lower(report_header) == 'training':
            file_name = upper(classifier_type) + "_TRAINING_REPORT "
            file_name = tra_dir + file_name + "(" + subject_name + ") " + feature_set + ".txt"

            if does_file_exist(file_name) is False:
                return create_training_report(subject_name=subject_name, classifier_type=classifier_type,
                                              feature_set=feature_set)
            else:
                return file_name
        elif lower(report_header) == 'classification':
            file_name = upper(classifier_type) + "_CLASSIFICATION_REPORT "
            file_name = cla_dir + file_name + "(" + subject_name + ") " + feature_set + ".txt"

            if does_file_exist(file_name) is False:
                return create_classification_report(subject_name=subject_name, classifier_type=classifier_type,
                                                    feature_set=feature_set)
            else:
                return file_name
        else:
            print("Invalid Report Heading")
    else:
        return file_name


def does_file_exist(file_name):
    exists = False
    if os.path.exists(file_name):
        exists = True

    return exists


def do_parameters_match(file_name, labels):
    reader = open(file_name, "r")

    first_line = reader.readline()

    if first_line is not None:
        file_labels = first_line.split(",")
        num_params = len(file_labels)

        if num_params == len(labels):
            i = 0
            while i < num_params:
                file_label = file_labels[i].strip()
                other_label = labels[i].strip()

                # Exit the method once a mismatch is found
                if not file_label == other_label:
                    print("-" + file_label.strip() + '-')
                    print('-' + other_label.strip() + '-')
                    print("Parameter mismatch")
                    return False

                i += 1
        else:
            print("Parameter size mismatch")
            return False
    else:
        print("Parameter line is empty")
        return False
    return True


# GESTURE DATABASE FUNCTIONS
def create_gesture_database(file_name):
    file_name = con_dir + file_name
    gestures = ["fist", "one", "two", "three", "four", "five"]

    writer = open(file_name, 'w')
    writer.close()

    writer = open(file_name, 'a')
    for gesture in gestures:
        writer.write(gesture + "\n")

    writer.close()


# Returns all file names inside current directory (or given directory if omitted) with matching extension
def get_data_files(directory=dat_dir, combined=False):
    extension = '.csv'
    data_file_names = []
    if combined is True:
        for file_name in os.listdir(com_dir):
            file_name = com_dir + file_name
            if file_name.endswith(extension):
                data_file_names.append(file_name)
    else:
        for file_name in os.listdir(directory):
            file_name = directory + file_name
            if file_name.endswith(extension):
                data_file_names.append(file_name)

    return data_file_names


def get_unseen_data_files(directory=uns_dir, combined=False):
    extension = '.csv'
    unseen_data_files = []
    if combined is True:
        for file_name in os.listdir(cun_dir):
            # print file_name
            file_name = cun_dir + file_name
            if file_name.endswith(extension):
                unseen_data_files.append(file_name)
    else:
        for file_name in os.listdir(directory):
            file_name = directory + file_name
            if file_name.endswith(extension):
                unseen_data_files.append(file_name)

    return unseen_data_files


def get_pickle_files(directory=trd_dir):
    extension = '.pickle'
    pickle_file_names = []

    for file_name in os.listdir(directory):
        file_name = directory + file_name
        if file_name.endswith(extension):
            pickle_file_names.append(file_name)

    return pickle_file_names


def read_row(file_name, index=0, delimiter=','):
    reader = open(file_name, 'r')

    lines = reader.readlines()
    row = lines[index]
    data_list = row.split(delimiter)

    return data_list


def read_col(file_name, index=0, delimiter=','):
    file_name = con_dir + file_name
    reader = open(file_name, 'r')

    lines = reader.readlines()

    data_list = []
    for line in lines:
        content = line.split(delimiter)
        data = content[index].strip()
        data_list.append(data)

    return data_list


def read_all(file_name):
    reader = open(file_name, 'r')

    lines = reader.readlines()

    return lines


def append_to_file(file_name, lines):
    writer = open(file_name, 'a')
    writer.write(lines)
    writer.write("\n")
    writer.close()
    pass


# Creating Excel Reports
def create_csv_results(file_name, labels):
    writer = open(file_name, 'w')
    writer.write(",".join(labels))
    writer.write('\n')
    writer.close()
    pass


def create_training_csv_results():
    file_name = tra_dir + "training results.csv"

    if does_file_exist(file_name) is False:
        labels = ['subject', 'classifier', 'gesture set', 'feature set', 'accuracy', 'time', 'accuracy penalty']
        create_csv_results(file_name=file_name, labels=labels)
        pass


def append_classification_csv_results(personalized, classifier_type, training_score, train_subject, test_subject,
                                      gesture_set, gesture, prediction, feature_set, correct, time):
    file_name = cla_dir + "classification results.csv"
    pg_file_name = cla_dir + "pg classification results.csv"

    labels = ['type', 'classifier', 'training score', 'trained subject', 'test subject', 'gesture set',
              'gesture', 'prediction', 'feature set', 'correct', 'time']
    pg_labels = ['gesture set', 'gesture', 'classifier', 'feature type', 'correct']

    if does_file_exist(file_name) is False:
        create_csv_results(file_name=file_name, labels=labels)
        pass

    if does_file_exist(pg_file_name) is False:
        create_csv_results(file_name=pg_file_name, labels=pg_labels)
        pass

    # Write on file_name
    writer = open(file_name, 'a')
    values = [personalized, classifier_type, str(training_score), train_subject, test_subject, feature_set,
              gesture_set, gesture, str(prediction), str(correct), str(time)]
    entry = ",".join(values)

    writer.write(entry)
    writer.write("\n")
    writer.close()

    # Write on pg file name
    writer = open(pg_file_name, 'a')
    values = [gesture_set, gesture, classifier_type, feature_set, correct]
    entry = ",".join(values)

    writer.write(entry)
    writer.write("\n")
    writer.close()


def append_lighting_csv_results(classifier_type, training_score, train_subject, test_subject, gesture_set,
                                feature_set, correct, total, accuracy):
    file_name = cla_dir + "lighting results.csv"
    labels = ['classifier', 'training score', 'trained subject', 'test subject', 'gesture set',
              'feature set', 'correct', 'total', 'accuracy']

    if does_file_exist(file_name) is False:
        create_csv_results(file_name=file_name, labels=labels)
        pass

    writer = open(file_name, 'a')
    values = [classifier_type, str(training_score), train_subject, test_subject, gesture_set, feature_set,
              str(correct), str(total), str(accuracy)]
    entry = ",".join(values)

    writer.write(entry)
    writer.write("\n")
    writer.close()


def append_repeatability_csv_results(classifier_type, training_score, train_subject, test_subject, gesture_set, gesture,
                                     feature_set, consecutive_correct, consecutive_incorrect):
    file_name = cla_dir + "repeatability results.csv"
    labels = ['classifier', 'training score', 'trained subject', 'test subject', 'gesture set', 'gesture',
              'feature set', 'consecutive_correct', 'consecutive_incorrect']

    if does_file_exist(file_name) is False:
        create_csv_results(file_name=file_name, labels=labels)
        pass

    writer = open(file_name, 'a')
    values = [classifier_type, str(training_score), train_subject, test_subject, gesture_set, gesture, feature_set,
              str(consecutive_correct), str(consecutive_incorrect)]
    entry = ",".join(values)

    writer.write(entry)
    writer.write("\n")
    writer.close()


def append_training_csv_results(subject, classifier_type, gesture_set, feature_set, accuracy, time, penalty_acc):
    file_name = tra_dir + "training results.csv"

    if does_file_exist(file_name) is False:
        create_training_csv_results()
        pass

    writer = open(file_name, 'a')
    values = [subject, classifier_type, gesture_set, feature_set, str(accuracy), str(time), str(penalty_acc)]
    entry = ",".join(values)

    writer.write(entry)
    writer.write("\n")
    writer.close()


def acquire_data_from_csv(csv_file):
    # Read csv file
    data = pd.read_csv(csv_file)
    X = np.array(data.drop(['class'], 1))
    y = np.array(data['class'])

    return X, y


def get_params():
    # Get all raw data
    data_files = get_data_files()
    # Get all trained data
    trained_data = get_pickle_files()

    # Get all unseen data
    unseen_data_files = get_unseen_data_files(combined=False)
    unseen_combined_files = get_unseen_data_files(combined=True)

    # Combined unseen data files
    all_unseen_data_files = []
    all_unseen_data_files.extend(unseen_data_files)
    all_unseen_data_files.extend(unseen_combined_files)

    # Get all types of features
    feature_set_list = [
        'finger-angle-and-palm-distance',
        'finger-angle-using-bones',
        'finger-between-distance',
        'finger-to-palm-distance',
    ]
    # Get all types of gestures
    gesture_set_list = [
        'COUNTING GESTURES',
        'STATUS GESTURES',
    ]
    subject_name_list = read_col("subjects.txt")

    return data_files, trained_data, unseen_data_files, unseen_combined_files, all_unseen_data_files, feature_set_list, gesture_set_list, subject_name_list
