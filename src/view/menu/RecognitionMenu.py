import time
from collections import Counter

from controller import LeapHandAcquisitor
from controller.LeapDataTrainer import SVM_Trainer, NN_Trainer, DT_Trainer
import controller.LeapIO as io
import controller.Printer as printer
from string import strip, lower
import Leap
from controller.LeapDataClassifier import LeapDataClassifier
from controller.LeapHandAcquisitor import LeapHandAcquisitor
import controller.LeapFeatureExtractor as extractor
import view.prompt.Prompter as prompter

def show(leap_controller):
    done = False
    while done is False:
        print("---------------------")
        print("REAL TIME RECOGNITION")
        print("---------------------")
        print("(1) - Live Demo")
        print("(2) - Loop Demo")
        print("(3) - Real Time")
        print("(0) - Exit Program")

        choice = raw_input("Your Choice: ")
        print("Your Input : " + choice)
        if choice == '1':

            sample_list = []
            hand_list = []

            # Show files available for classification (pickle files)
            list_data_files = io.get_pickle_files()
            print("* List of Pickle Files *")
            printer.print_numbered_list(list_data_files)
            print("\n")

            choice = raw_input("Enter the pickle file for classifier \nPickle Index  : ")
            chosen_pickle = list_data_files[int(choice) - 1]

            # Get File Path without folders
            file_path = chosen_pickle.split("\\")[-1]
            # Get parameters
            classifier_type = file_path.split(" ")[0]
            gesture_set = strip(file_path.split("--")[0].split(")")[1])
            feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]

            chosen_pickle_no_extension = chosen_pickle.rsplit(".", 1)[0]
            kernel_type = chosen_pickle_no_extension.rsplit("_", 1)[-1]
            subject_name = chosen_pickle_no_extension.rsplit("(", 1)[1].rsplit(")")[0]
            feature_type = strip(chosen_pickle_no_extension.rsplit(")")[1].rsplit(".")[0])

            acquisitor = LeapHandAcquisitor(leap_controller=leap_controller)
            classifier_controller = LeapDataClassifier(acquisitor=acquisitor)

            # Do demo for 1 minute
            time_elapsed = 0.0
            stack = 0
            print("\r.."),
            start_time = time.time()
            while time_elapsed <= 60:
                time.sleep(0.1)
                time_elapsed = round(time.time() - start_time, 2)

                frame = leap_controller.frame()
                hands = frame.hands
                trainer = None
                if len(hands) > 0:
                    hand = hands[0]
                    print("\n" + feature_set)

                    # Obtain classifier type
                    if lower(classifier_type) == 'nn':
                        # Get set hyper parameters
                        activation = file_path.split(".")[0].split("--")[1].split("_")[1]
                        # Get NN Trainer
                        trainer = NN_Trainer(subject_name=subject_name, feature_type=feature_set, activation=activation,
                                             gesture_set=gesture_set)
                        trainer.load(pickle_name=file_path)
                        pass
                    elif lower(classifier_type) == 'svm':
                        # Get set hyper parameters
                        kernel_type = file_path.split(".")[0].split("--")[1].split("_")[1]
                        # Get SVM Trainer
                        trainer = SVM_Trainer(subject_name=subject_name, feature_type=feature_set,
                                              kernel_type=kernel_type,
                                              gesture_set=gesture_set)
                        trainer.load(pickle_name=file_path)
                    elif lower(classifier_type) == 'dt':
                        # Get set hyper parameters
                        criterion_type = file_path.split(".")[0].split("--")[1].split("_")[1]
                        # Get NN Trainer
                        trainer = DT_Trainer(subject_name=subject_name, feature_type=feature_set,
                                             criterion_type=criterion_type,
                                             gesture_set=gesture_set)
                        trainer.load(pickle_name=file_path)

                    # Predict
                    data = get_relevant_data(feature_set=feature_set, hand=hand)
                    X_data = []
                    for leap_data in data:
                        X_data.append(leap_data.value)

                    prediction = trainer.classify(X=[X_data])
                    print("\rTime Elapsed : " + str(time_elapsed) + " seconds ---> Prediction : " + str(prediction[0])),

                    # while prediction == trainer.classify(get_relevant_data(kernel_type=kernel_type, file_name=chosen_pickle_no_extension, acquisitor=acquisitor, hand=hand)) and stack < 5:
                    #     hand = leap_controller.frame().hands[0]
                    #     stack += 1
                    #
                    # if stack >= 5:
                    #     print("\rTime Elapsed : " + str(time_elapsed) + " seconds ---> Prediction : " + str(prediction[0])),
                    #     stack = 0

                else:
                    print("\rTime Elapsed : " + str(time_elapsed) + " seconds ---> Prediction : asdaNone"),
                    stack = 0

            print("System       : Demo has completed.\n\n")
            print("----------------------------------------------------")
            print("Number of Samples          : " + str(len(sample_list)))
            print("Number of Frame per Sample : " + str(len(hand_list)))
            print("----------------------------------------------------\n\n")

            pass
        elif choice == '2':
            done = True
            up = Leap.Vector.up
            print("")
            while done is True:
                frame = leap_controller.frame()
                hand = frame.hands[0]

                test = hand.palm_normal.dot(up)

                x_d = round(hand.palm_normal.x, 5)
                y_d = round(hand.palm_normal.y, 5)
                z_d = round(hand.palm_normal.z, 5)
                # print("\r" + str(x_d) + ", " + str(y_d) + ", " + str(z_d)),
                print test
            pass
        elif choice == '3':
            real_time(leap_controller=leap_controller)
            pass
        elif choice == '0':
            # Exit
            done = True
            pass


def real_time(leap_controller):
    acquisitor = LeapHandAcquisitor(leap_controller=leap_controller)
    classifier_controller = LeapDataClassifier(acquisitor=acquisitor)
    # Obtain relevant gesture information and list
    gesture_set, gesture_list, _ = prompter.prompt_gesture_set()
    # Obtain all other relevant data
    _, trained_data_files, _, _, _, feature_set_list, _, _ = io.get_params()
    test_subject = prompter.prompt_subject_name()

    # Number of sample group per gesture
    intervals = 50
    hand_configurations = ['LEFT HAND', 'RIGHT HAND']

    # Initialize prediction dict for ensemble
    prediction_dict_ensemble = {}
    classifier_types = ['Support Vector Machine', 'Multi-Layer Perceptron Neural Network', 'Decision Tree']
    for classifier_type in classifier_types:
        prediction_dict_ensemble[classifier_type] = []

    # Initialize correct predictions dictionaries and list of trainer
    trainer_list = []
    gesture_dict = {}
    trained_data_dict = {}
    for trained_data in trained_data_files:
        # Get File Path without folders
        file_path = trained_data.split("\\")[-1]
        # Get parameters
        classifier_type = file_path.split(" ")[0]
        gesture_set = strip(file_path.split("--")[0].split(")")[1])
        feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
        train_subject = file_path.split("(")[1].split(")")[0]

        # Obtain learning model
        trainer = classifier_controller.obtain_classifier(classifier_type=classifier_type,
                                                                   gesture_set=gesture_set,
                                                                   feature_set=feature_set,
                                                                   train_subject=train_subject,
                                                                   pickle_file=trained_data)
        trainer_list.append(trainer)

        # Each gesture corresponds to an index of the gesture data list results
        gesture_ind = 0
        for gesture in gesture_list:
            gesture_dict[gesture] = gesture_ind
            gesture_ind += 1

        # Initialize each gesture data list - contains occurrences
        gesture_data_list = []
        for g in range(len(gesture_list)):
            gesture_data_list.append(0)
            g += 1

        # Add initialized gesture dictionary to trained data dictionary
        trained_data_dict[trainer] = gesture_data_list

        print file_path

    # Number of samples per group
    sample_size = 20
    time_elapsed = 0.0
    start_time = time.time()
    while time_elapsed <= 60:
        time_elapsed = round(time.time() - start_time, 2)
        for i in range(intervals):
            stack = 0
            sample_obtained = False
            feature_map_list = []
            while sample_obtained is False:
                # Obtain hand
                hand = acquisitor.acquire_single_hand_data(supervised=False)
                # Ensure hand is not None
                if hand is not None:
                    feature_map = extractor.extract_all_feature_type(hand=hand)
                    feature_map_list.append(feature_map)

                    stack += 1

                    if stack >= sample_size:
                        sample_obtained = True
                        pass
                # Reset
                else:
                    stack = 0
                    for trainer in trainer_list:
                        print("\rPrediction -- None"),
                        # For each gesture, reset dictionary
                        for g in range(len(gesture_list)):
                            trained_data_dict[trainer][g] = 0
                    pass
                # print("\r\rStack: " + str(stack)),
                time.sleep(0.005)

            # Check for each dictionary
            for trainer in trainer_list:
                for feature_map in feature_map_list:
                    for feature_pair in feature_map:
                        feature_name = feature_pair[0]
                        feature_data_set = feature_pair[1]

                        if trainer.feature_type == feature_name:
                            prediction = classifier_controller.do_classification_from_features(
                                trainer=trainer,
                                feature_data_set=feature_data_set,
                            )
                            trained_data_dict[trainer][gesture_dict[prediction]] += 1

            # For each learning model, find the most occurring prediction in the sample window
            prediction_list = []
            for trainer in trainer_list:
                # Get the gesture with highest value in sample window
                result_gesture_data_list = trained_data_dict[trainer]
                max_value = result_gesture_data_list.index(max(result_gesture_data_list))
                for key, value in gesture_dict.items():
                    if max_value == value:
                        prediction = key
                        io.append_sampling_csv_report(test_subject=test_subject,
                                                      trainer=trainer,
                                                      result_list=result_gesture_data_list,
                                                      gesture=gesture,
                                                      prediction=prediction
                                                      )
                        prediction_list.append(prediction)
                        prediction_dict_ensemble[trainer.classifier_name].append(prediction)

            # print("\r" + str(prediction_list))

            # Find the most common prediction for each ensemble or learning group
            classifier_type = 'Multi-Layer Perceptron Neural Network'
            ensemble_prediction_list = prediction_dict_ensemble[classifier_type]
            occurrence_count = Counter(ensemble_prediction_list)
            occurrence_count.most_common()
            final_prediction = occurrence_count.most_common(1)[0][0]

            print("\rPrediction -- " + final_prediction),

            # Reset ensemble dictionary
            prediction_dict_ensemble[classifier_type] = []

            # Rest sampling dictionary
            for trainer in trainer_list:
                # For each gesture, reset dictionary
                for g in range(len(gesture_list)):
                    trained_data_dict[trainer][g] = 0
            i += 1



def get_relevant_data(feature_set, hand=None):
    data = []
    if "finger-to-palm-distance" == feature_set:
        _, data = extractor.extract_finger_palm_distance(hand=hand)
    elif "finger-angle-using-bones" == feature_set:
        _, data = extractor.extract_finger_palm_angle(hand=hand)
    elif "finger-angle-and-palm-distance" == feature_set:
        _, data = extractor.extract_finger_palm_angle_distance(hand=hand)
    elif "finger-between-distance" == feature_set:
        _, data = extractor.extract_finger_finger_distance(hand=hand)

    return data