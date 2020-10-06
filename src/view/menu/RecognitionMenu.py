import time
from controller.LeapDataTrainer import SVM_Trainer, NN_Trainer, DT_Trainer
import controller.LeapIO as io
import controller.Printer as printer
from string import strip, lower
import Leap
from controller.LeapDataClassifier import LeapDataClassifier
import controller.LeapHandAcquisitor as acquisitor
import controller.LeapFeatureExtractor as extractor

def show(leap_controller):
    done = False
    while done is False:
        print("---------------------")
        print("REAL TIME RECOGNITION")
        print("---------------------")
        print("(1) - Live Demo")
        print("(2) - Loop Demo")
        print("(9) - One Gesture Test")
        print("(0) - Exit Program")

        choice = raw_input("Your Choice: ")
        print("Your Input : " + choice)
        if choice == '1':
            start_time = time.time()

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

            classifier_controller = LeapDataClassifier(acquisitor=acquisitor)

            # Do demo for 1 minute
            time_elapsed = 0.0
            stack = 0
            print("\r.."),
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
        elif choice == '9':
            pass
        elif choice == '0':
            # Exit
            done = True
            pass



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