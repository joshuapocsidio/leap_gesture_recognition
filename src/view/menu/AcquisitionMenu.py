# Controller libraries
import controller.LeapIO as io
import controller.LeapFeatureExtractor as extractor

# View Libraries
import view.prompt.Prompter as prompter

# Built in libraries
import time
from string import upper


def show(acquisitor):
    # Shows menu for data acquisition related menu
    done = False
    while done is False:
        print("----------------")
        print("DATA ACQUISITION")
        print("----------------")
        print("(1) - Manual Acquisition : 100 Data")
        print("(2) - Systematic Acquisition : Training Set")
        print("(3) - Systematic Acquisition : Testing Set")
        print("(0) - Back")

        choice = raw_input("Your Choice: ")

        if choice == '1':
            do_manual_acquisition(acquisitor=acquisitor)
            pass
        elif choice == '2':
            do_systematic_acquisition(acquisitor=acquisitor, training=True)
            # do_acquisition(controller=controller, intervals=100, checkpoints=10)
            pass
        elif choice == '3':
            do_systematic_acquisition(acquisitor=acquisitor, testing=False)
            # do_acquisition(controller=controller, intervals=25, checkpoints=5)
            pass
        elif choice == '0':
            done = True
            pass
        else:
            print("Please try again.")
            pass
    pass


def do_manual_acquisition(acquisitor):
    done = False
    while done is False:
        print("----------------")
        print("DATA ACQUISITION")
        print("----------------")
        print("(1) - Feature Data Set : Finger to Palm Distance")
        print("(2) - Feature Data Set : Finger to Palm Angle")
        print("(3) - Feature Data Set : Finger to Palm Distance and Angle")
        print("(4) - Feature Data Set : Finger to Finger Distance")
        print("(9) - Feature Data Set : ALL")
        print("(0) - Back")

        choice = raw_input("Your Choice: ")

        # First prompt for subject name
        if choice.isdigit():
            num_choice = int(choice)
            if 0 < num_choice <= 9:

                # Prompt relevant information
                subject_name = prompter.prompt_subject_name()
                gesture_set, _, gesture_src = prompter.prompt_gesture_set()
                gesture_name = prompter.prompt_gesture_name(gesture_src)
                iterations = prompter.prompt_iterations()

                # Acquiring hand set
                hand_set = acquisitor.acquire_multiple_hand_data(iterations=iterations, intervals=iterations)
                # Actual menu actions
                if choice == '1':
                    for hand in hand_set:
                        feature_name, feature_data_set = extractor.extract_finger_palm_distance(hand)
                        file_name = "(" + gesture_set + ") --" + feature_name
                        io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=gesture_name,
                                     data_set=feature_data_set)
                        pass
                elif choice == '2':
                    for hand in hand_set:
                        feature_name, feature_data_set = extractor.extract_finger_palm_angle(hand)
                        file_name = "(" + gesture_set + ") --" + feature_name
                        io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=gesture_name,
                                     data_set=feature_data_set)
                        pass
                elif choice == '3':
                    for hand in hand_set:
                        feature_name, feature_data_set = extractor.extract_finger_palm_angle_distance(hand)
                        file_name = "(" + gesture_set + ") --" + feature_name
                        io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=gesture_name,
                                     data_set=feature_data_set)
                        pass
                elif choice == '4':
                    for hand in hand_set:
                        feature_name, feature_data_set = extractor.extract_finger_finger_distance(hand)
                        file_name = "(" + gesture_set + ") --" + feature_name
                        io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=gesture_name,
                                     data_set=feature_data_set)
                        pass
                elif choice == '9':
                    for hand in hand_set:
                        feature_name, feature_data_set = extractor.extract_all_feature_type(hand)
                        file_name = "(" + gesture_set + ") --" + feature_name
                        io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=gesture_name,
                                     data_set=feature_data_set)
                        pass
                elif choice == '0':
                    done = True
                    pass
                else:
                    print("Please try again - choice not recognized")
                    pass
            else:
                print("Please try again - out of bounds")
                pass
        else:
            print("Please try again - input is not a digit")
            pass
    pass


def do_systematic_acquisition(acquisitor, training=False, testing=False):
    # Pre-defined gesture titles
    hand_config = ['LEFT HAND', 'RIGHT HAND']

    if training is True:
        intervals = 100
        checkpoints = 10
        pass
    elif testing is True:
        intervals = 25
        checkpoints = 5
        pass
    else:
        intervals = 10
        checkpoints = 1
        pass

    # Prompt relevant variables
    subject_name = prompter.prompt_subject_name()

    gesture_set, gesture_list, _ = prompter.prompt_gesture_set()

    # Loop between gesture
    i_ges = 0
    hand_set = []
    feature_map = []
    while i_ges < len(gesture_list):
        cur_gesture = gesture_list[i_ges]

        # Loop between hands
        i_hand = 0
        while i_hand < len(hand_config):
            print("Acquiring Gesture : " + upper(cur_gesture) + " --> " + hand_config[i_hand] + " ")

            # Loop between each gesture data taken
            print("\rProgress ----> " + str(0) + "/" + str(intervals) + " acquired"),
            raw_input("\nSystem       :       Press any key to get data: "),
            n_taken = 0

            while n_taken < intervals:
                time.sleep(0.05)
                # Acquire data and append to hand set
                hand = None
                while hand is None:
                    hand = acquisitor.acquire_single_hand_data()

                hand_set.append(hand)

                # Obtain all the feature name and data set
                feature_map = extractor.extract_all_feature_type(hand=hand)
                for feature_pair in feature_map:
                    feature_name = feature_pair[0]
                    feature_data_set = feature_pair[1]

                    # Create the file name
                    file_name = gesture_set + "--" + feature_name
                    # Save data based on feature name, feature data set, and file name
                    io.save_data(file_name=file_name, subject_name=subject_name, gesture_name=cur_gesture,
                                 data_set=feature_data_set)

                n_taken += 1
                print("\rProgress ----> " + str(n_taken) + "/" + str(intervals) + " acquired"),

                if n_taken % checkpoints == 0:
                    if n_taken == intervals:
                        raw_input("\nSystem       :       Gesture Checkpoint reached. Press any key to continue"),
                    else:
                        raw_input("\nSystem       :       Press any key to get data: "),
                pass
            print(" -- SUCCESS!\n")
            i_hand += 1
            pass

        i_ges += 1
        pass
