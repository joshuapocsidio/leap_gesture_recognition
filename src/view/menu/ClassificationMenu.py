# Import built in libraries
import time
from string import lower, upper, strip

# Import Controller libraries
import controller.LeapIO as io
import controller.Printer as printer
from controller.LeapDataTrainer import SVM_Trainer

# Import View Libraries
import view.prompt.Prompter as prompter


class ClassificationMenu:
    def __init__(self, classification_controller):
        self.classification_controller = classification_controller

    def show(self):
        # Shows menu for classifying gesture data
        done = False

        while done is False:
            print("-------------------")
            print("DATA CLASSIFICATION")
            print("-------------------")
            print("(1) - Unseen Data Classification Test")
            print("(2) - Repeatability Classification Test")
            print("(3) - Lighting Variance Classification Test")
            print("(0) - Back")

            choice = raw_input("Your Choice: ")

            if choice == '1':
                self.unseen_data_classification_test()
                pass
            elif choice == '2':
                self.repeatability_classification_test()
                pass
            elif choice == '3':
                self.lighting_classification_test()
                pass
            elif choice == '0':
                done = True
                pass

    def unseen_data_classification_test(self):
        done = False
        while done is False:
            print("--------------------------")
            print("UNSEEN DATA CLASSIFICATION")
            print("--------------------------")
            print("(1) - Single Subject Personalized Test")
            print("(2) - Single Subject Non-Personalized Test")
            print("(3) - Multi Subject Personalized Test")
            print("(4) - Multi Subject Non-Personalized Test")
            print("(5) - Full Systematic Test")
            print("(0) - Back")

            choice = raw_input("Your Choice: ")

            file_name = None
            _, trained_pickle_list, _, _, unseen_data_list, _, _, subject_name_list = io.get_params()
            # PERSONALIZED TEST - one trained subject for own unseen data
            if choice == '1':
                choice_subject_name = prompter.prompt_subject_name()

                matching_pickle_list = []
                # Obtain all trained data for this test subject
                for trained_pickle in trained_pickle_list:
                    # Obtain the subject of trained data for each matching pickle file
                    trained_subject = trained_pickle.rsplit("(", 1)[1].rsplit(")")[0]

                    # Want all pickle files for this subject
                    if lower(trained_subject) == lower(choice_subject_name):
                        print(trained_pickle)
                        matching_pickle_list.append(trained_pickle)
                        pass

                # Get Own Unseen Data
                for unseen_data in unseen_data_list:
                    # Obtain feature set
                    unseen_feature_set = unseen_data.split("--")[1].split(".csv")[0]
                    # Obtain name since personalized test - subject name matters
                    unseen_subject_name = unseen_data.split("(")[1].split(")")[0]
                    # Obtain gesture set since only want to compare matching gestures
                    unseen_gesture_set = strip(unseen_data.split("--")[0].split(")")[1])

                    # Check for all matching trained data from chosen name
                    for matching_pickle in matching_pickle_list:
                        # Get File Path without folders
                        file_path = matching_pickle.split("\\")[-1]
                        # Get parameters
                        classifier_type = file_path.split(" ")[0]
                        gesture_set = strip(file_path.split("--")[0].split(")")[1])
                        feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
                        # Only do classification if same type of feature set ---> Otherwise will not work at all
                        if unseen_feature_set == feature_set and \
                                upper(unseen_subject_name) == upper(choice_subject_name) and \
                                unseen_gesture_set == gesture_set:
                            # Do classification from csv
                            file_name = self.classification_controller.do_classification_from_csv(
                                pickle_file=matching_pickle,
                                train_subject=choice_subject_name,
                                test_subject=unseen_subject_name,
                                classifier_type=classifier_type,
                                gesture_set=gesture_set,
                                feature_set=feature_set,
                                unseen_data=unseen_data,
                                file_name=file_name
                            )

            # NON-PERSONALIZED TEST - one trained subject for others' data
            elif choice == '2':
                choice_subject_name = prompter.prompt_subject_name()

                matching_pickle_list = []
                # Obtain all trained data for this test subject
                for trained_pickle in trained_pickle_list:
                    # Obtain the subject of trained data for each matching pickle file
                    trained_subject = trained_pickle.rsplit("(", 1)[1].rsplit(")")[0]

                    # Want all pickle files for this subject
                    if lower(trained_subject) == lower(choice_subject_name):
                        print(trained_pickle)
                        matching_pickle_list.append(trained_pickle)
                        pass

                # Get Others' Unseen Data
                for unseen_data in unseen_data_list:
                    # Obtain feature set
                    unseen_feature_set = unseen_data.split("--")[1].split(".csv")[0]
                    # Obtain name since personalized test - subject name matters
                    unseen_subject_name = unseen_data.split("(")[1].split(")")[0]
                    # Obtain gesture set since only want to compare matching gestures
                    unseen_gesture_set = strip(unseen_data.split("--")[0].split(")")[1])

                    # Check for all matching trained data from chosen name
                    for matching_pickle in matching_pickle_list:
                        # Get File Path without folders
                        file_path = matching_pickle.split("\\")[-1]
                        # Get parameters
                        classifier_type = file_path.split(" ")[0]
                        gesture_set = strip(file_path.split("--")[0].split(")")[1])
                        feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]

                        # Only do classification if same type of feature set ---> Otherwise will not work at all
                        if unseen_feature_set == feature_set and \
                                upper(unseen_subject_name) != upper(choice_subject_name) and \
                                unseen_gesture_set == gesture_set:
                            # Do classification from csv
                            self.classification_controller.do_classification_from_csv(
                                pickle_file=matching_pickle,
                                train_subject=choice_subject_name,
                                test_subject=unseen_subject_name,
                                classifier_type=classifier_type,
                                gesture_set=gesture_set,
                                feature_set=feature_set,
                                unseen_data=unseen_data,
                                file_name=file_name
                            )

            elif choice == '3':
                matching_pickle_list = []
                for subject_name in subject_name_list:
                    # Obtain all trained data for this test subject
                    for trained_pickle in trained_pickle_list:
                        # Obtain the subject of trained data for each matching pickle file
                        trained_subject = trained_pickle.rsplit("(", 1)[1].rsplit(")")[0]

                        # Want all pickle files for this subject
                        if lower(trained_subject) == lower(subject_name):
                            print(trained_pickle)
                            matching_pickle_list.append(trained_pickle)
                            pass

                    # Get Own Unseen Data
                    for unseen_data in unseen_data_list:
                        # Obtain feature set
                        unseen_feature_set = unseen_data.split("--")[1].split(".csv")[0]
                        # Obtain name since personalized test - subject name matters
                        unseen_subject_name = unseen_data.split("(")[1].split(")")[0]
                        # Obtain gesture set since only want to compare matching gestures
                        unseen_gesture_set = strip(unseen_data.split("--")[0].split(")")[1])

                        # Check for all matching trained data from chosen name
                        for matching_pickle in matching_pickle_list:
                            # Get File Path without folders
                            file_path = matching_pickle.split("\\")[-1]
                            # Get parameters
                            classifier_type = file_path.split(" ")[0]
                            gesture_set = strip(file_path.split("--")[0].split(")")[1])
                            feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]

                            # Only do classification if same type of feature set ---> Otherwise will not work at all
                            if unseen_feature_set == feature_set and \
                                    unseen_subject_name == subject_name and \
                                    upper(subject_name) != "COMBINED SUBJECTS" and \
                                    unseen_gesture_set == gesture_set:
                                # Do classification from csv
                                self.classification_controller.do_classification_from_csv(
                                    pickle_file=matching_pickle,
                                    train_subject=subject_name,
                                    test_subject=unseen_subject_name,
                                    classifier_type=classifier_type,
                                    gesture_set=gesture_set,
                                    feature_set=feature_set,
                                    unseen_data=unseen_data,
                                    file_name=file_name
                                )

            elif choice == '4':
                matching_pickle_list = []
                for subject_name in subject_name_list:
                    # Obtain all trained data for this test subject
                    for trained_pickle in trained_pickle_list:
                        # Obtain the subject of trained data for each matching pickle file
                        trained_subject = trained_pickle.rsplit("(", 1)[1].rsplit(")")[0]

                        # Want all pickle files for this subject
                        if lower(trained_subject) == lower(subject_name):
                            print(trained_pickle)
                            matching_pickle_list.append(trained_pickle)
                            pass

                    # Get Own Unseen Data
                    for unseen_data in unseen_data_list:
                        # Obtain feature set
                        unseen_feature_set = unseen_data.split("--")[1].split(".csv")[0]
                        # Obtain name since personalized test - subject name matters
                        unseen_subject_name = unseen_data.split("(")[1].split(")")[0]
                        # Obtain gesture set since only want to compare matching gestures
                        unseen_gesture_set = strip(unseen_data.split("--")[0].split(")")[1])

                        # Check for all matching trained data from chosen name
                        for matching_pickle in matching_pickle_list:
                            # Get File Path without folders
                            file_path = matching_pickle.split("\\")[-1]
                            # Get parameters
                            classifier_type = file_path.split(" ")[0]
                            gesture_set = strip(file_path.split("--")[0].split(")")[1])
                            feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]

                            # Only do classification if same type of feature set ---> Otherwise will not work at all
                            if unseen_feature_set == feature_set and \
                                    unseen_subject_name != subject_name and \
                                    unseen_gesture_set == gesture_set:
                                # Do classification from csv
                                self.classification_controller.do_classification_from_csv(
                                    pickle_file=matching_pickle,
                                    train_subject=subject_name,
                                    test_subject=unseen_subject_name,
                                    classifier_type=classifier_type,
                                    gesture_set=gesture_set,
                                    feature_set=feature_set,
                                    unseen_data=unseen_data,
                                    file_name=file_name,
                                )

            elif choice == '5':
                matching_pickle_list = []
                total = len(subject_name_list) * ((len(subject_name_list) * 8) + (len(subject_name_list) * 4) + 12)
                num = 0
                u_ind = 1
                # Get Own Unseen Data
                for unseen_data in unseen_data_list:
                    # Obtain feature set
                    unseen_feature_set = unseen_data.split("--")[1].split(".csv")[0]
                    # Obtain name since personalized test - subject name matters
                    unseen_subject_name = unseen_data.split("(")[1].split(")")[0]
                    # Obtain gesture set since only want to compare matching gestures
                    unseen_gesture_set = strip(unseen_data.split("--")[0].split(")")[1])

                    # Check for all matching trained data from chosen name
                    t_ind = 1
                    for trained_pickle in trained_pickle_list:
                        # Get File Path without folders
                        file_path = trained_pickle.split("\\")[-1]
                        # Get parameters
                        classifier_type = file_path.split(" ")[0]
                        gesture_set = strip(file_path.split("--")[0].split(")")[1])
                        feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
                        subject_name = file_path.split("(")[1].split(")")[0]

                        # Only do classification if same type of feature set ---> Otherwise will not work at all
                        if unseen_feature_set == feature_set and \
                                unseen_gesture_set == gesture_set:
                            params = trained_pickle.split("_")[-1].split(".")[0]

                            # Do classification from csv
                            print("\n"),
                            accuracy = self.classification_controller.do_classification_from_csv(
                                pickle_file=trained_pickle,
                                train_subject=subject_name,
                                test_subject=unseen_subject_name,
                                classifier_type=classifier_type,
                                gesture_set=gesture_set,
                                feature_set=feature_set,
                                unseen_data=unseen_data,
                                file_name=file_name
                            )

                            print("\rUnseen Data (" + str(u_ind) + "\\" + str(len(unseen_data_list)) + ") :"
                                  + " Trained Data (" + str(t_ind) + "\\" + str(len(trained_pickle_list)) + ") :"
                                  + " ---> Unseen Data  : (" + unseen_subject_name + ") " + unseen_gesture_set + "--" + unseen_feature_set
                                  + " ---> Trained Data : (" + subject_name + ") " + gesture_set + "--" + feature_set
                                  + " # Classifier : " + classifier_type + ", Params : " + params
                                  + " ---> ACCURACY = " + str(accuracy * 100.0) + "%"),
                            num += 1
                        t_ind += 1
                    u_ind += 1
                print("")

            elif choice == '0':
                done = True
                pass
        pass

    def lighting_classification_test(self):
        total = 0
        intervals = 50
        # Obtain relevant gesture information and list
        gesture_set, gesture_list, _ = prompter.prompt_gesture_set()
        # Obtain all other relevant data
        _, trained_data_files, _, _, _, feature_set_list, _, _ = io.get_params()
        # Prompt for subject name
        test_subject = prompter.prompt_subject_name()

        # Initialize map of accuracies and corresponding trained data
        correct_map = {}
        for trained_data in trained_data_files:
            # Each slot is an integer representing correct number of predictions
            correct_map[trained_data] = 0

        for gesture in gesture_list:
            print("\rProgress ----> " + str(0) + "/" + str(0) + " acquired"),
            raw_input("\nGesture Stage - " + gesture + ". Press any key to start test."),
            for i in range(intervals):
                timeout = 10
                timeout_counter = 0
                hand = None

                while timeout_counter < timeout and hand is None:
                    hand = self.classification_controller.acquisitor.acquire_single_hand_data()
                    time.sleep(1)
                    timeout_counter += 1

                if timeout_counter >= timeout:
                    print("\n")
                    print("* * * NOTICE * * *")
                    print("Please reconnect the controller. Iterations completed -- " + str(i) + " iterations")
                    print("Returning to Classification Menu...\n")
                    return None

                # Checkpoint each iteration
                raw_input("\rProgress ----> " + str(i) + "/" + str(intervals) + " acquired. Press any key to continue"),

                train_c = 0
                # For each gesture, classify with all pickle files
                for trained_data in trained_data_files:
                    # Get File Path without folders
                    file_path = trained_data.split("\\")[-1]
                    # Get parameters
                    classifier_type = file_path.split(" ")[0]
                    gesture_set = strip(file_path.split("--")[0].split(")")[1])
                    feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
                    train_subject = file_path.split("(")[1].split(")")[0]

                    # Do classification and obtain results and iterate for 100
                    prediction, result, trainer = self.classification_controller.do_classification_from_hand(
                        pickle_file=trained_data,
                        train_subject=train_subject,
                        classifier_type=classifier_type,
                        feature_set=feature_set,
                        gesture_set=gesture_set,
                        chosen_gesture=gesture,
                        hand=hand,
                    )

                    # Check if prediction is correct
                    if prediction == gesture:
                        correct_map[trained_data] += 1

                total += 1
                i += 1

        for trained_data in trained_data_files:
            # Get File Path without folders
            file_path = trained_data.split("\\")[-1]
            # Get parameters
            classifier_type = file_path.split(" ")[0]
            gesture_set = strip(file_path.split("--")[0].split(")")[1])
            feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
            train_subject = file_path.split("(")[1].split(")")[0]

            # Obtain classifier type
            trainer = self.classification_controller.obtain_classifier(
                classifier_type=classifier_type,
                gesture_set=gesture_set,
                feature_set=feature_set,
                train_subject=train_subject,
                pickle_file=trained_data
            )

            # Save number of consecutive nums once consecutive streak is broken
            io.append_lighting_csv_results(
                classifier_type=classifier_type,
                training_score=round(trainer.training_acc * 100, 5),
                train_subject=trainer.subject_name,
                test_subject=test_subject,
                gesture_set=gesture_set,
                feature_set=feature_set,
                correct=str(correct_map[trained_data]),
                total=str(total),
                accuracy=str(round(float(correct_map[trained_data]) / float(total), 3)),
            )

    def repeatability_classification_test(self):
        intervals = 100
        # Obtain relevant gesture information and list
        gesture_set, gesture_list, _ = prompter.prompt_gesture_set()
        # Obtain all other relevant data
        _, trained_data_files, _, _, _, feature_set_list, _, _ = io.get_params()
        # Prompt for subject name
        test_subject = prompter.prompt_subject_name()

        # Initialize map of consecutive and corresponding trained data
        correct_map = {}
        incorrect_map = {}
        for trained_data in trained_data_files:
            # Each slot is a list of consecutive
            correct_map[trained_data] = []
            correct_map[trained_data].append(0)
            incorrect_map[trained_data] = []
            incorrect_map[trained_data].append(0)

        print("Consecutive Map : Initialised\n")
        # For each gesture, iteration i times
        for gesture in gesture_list:
            print("\rProgress ----> " + str(0) + "/" + str(0) + " acquired"),
            raw_input("\nGesture Stage - " + gesture + ". Press any key to start test."),

            for i in range(intervals):
                timeout = 10
                timeout_counter = 0
                hand = None

                while timeout_counter < timeout and hand is None:
                    hand = self.classification_controller.acquisitor.acquire_single_hand_data()
                    time.sleep(1)
                    timeout_counter += 1

                if timeout_counter >= timeout and hand is None:
                    print("\n")
                    print("* * * NOTICE * * *")
                    print("Please reconnect the controller. Iterations completed -- " + str(i) + " iterations")
                    print("Returning to Classification Menu...\n")
                    return None

                # Checkpoint each iteration
                raw_input("\rProgress ----> " + str(i) + "/" + str(intervals) + " acquired. Press any key to continue"),

                # For each gesture, classify with all pickle files
                for trained_data in trained_data_files:
                    # Get File Path without folders
                    file_path = trained_data.split("\\")[-1]
                    # Get parameters
                    classifier_type = file_path.split(" ")[0]
                    gesture_set = strip(file_path.split("--")[0].split(")")[1])
                    feature_set = file_path.split("--")[1].split(".pickle")[0].split("_")[0]
                    train_subject = file_path.split("(")[1].split(")")[0]

                    # Do classification and obtain results and iterate for 100
                    prediction, result, trainer = self.classification_controller.do_classification_from_hand(
                        pickle_file=trained_data,
                        train_subject=train_subject,
                        classifier_type=classifier_type,
                        feature_set=feature_set,
                        gesture_set=gesture_set,
                        chosen_gesture=gesture,
                        hand=hand,
                    )

                    if prediction == gesture:
                        # Update correct map
                        correct_map[trained_data][-1] += 1
                        # Save number of consecutive nums once consecutive streak is broken
                        if incorrect_map[trained_data][-1] > 0:
                            io.append_repeatability_csv_results(
                                classifier_type=classifier_type,
                                training_score=round(trainer.training_acc * 100, 5),
                                train_subject=trainer.subject_name,
                                test_subject=test_subject,
                                gesture_set=gesture_set,
                                gesture=gesture,
                                feature_set=feature_set,
                                consecutive_correct=0,
                                consecutive_incorrect=incorrect_map[trained_data[-1]]
                            )

                        # Reset incorrect streak
                        incorrect_map[trained_data].append(0)
                    else:
                        # Update incorrect map
                        incorrect_map[trained_data][-1] += 1
                        # Save number of consecutive nums once consecutive streak is broken
                        if correct_map[trained_data][-1] > 0:
                            io.append_repeatability_csv_results(
                                classifier_type=classifier_type,
                                training_score=round(trainer.training_acc * 100, 5),
                                train_subject=trainer.subject_name,
                                test_subject=test_subject,
                                gesture_set=gesture_set,
                                gesture=gesture,
                                feature_set=feature_set,
                                consecutive_correct=correct_map[trained_data][-1],
                                consecutive_incorrect=0
                            )

                        # Reset correct streak
                        correct_map[trained_data].append(0)
                i += 1
            print("\n Gesture - " + gesture + " stage >> Successful")

    def show_current(self, data_file, classifier_type, feature_set, gesture_set):
        print("Data File >> " + data_file)
        print("    Classifier  - " + classifier_type)
        print("    Feature Set - " + feature_set)
        print("    Gesture Set - " + gesture_set)

    ''' MIGHT REMOVE THIS
    def single_feature_classification(self):
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
        trained_subject = file_path.split("(")[1].split(")")[0]

        # Prompt user for gesture name
        gesture_set, _, gesture_src = prompter.prompt_gesture_set()
        chosen_gesture = prompter.prompt_gesture_name(gesture_src)

        # Obtain subject name
        subject_name = self.classification_controller.subject_name

        # Do Classification TODO : Fix this for future tests
        # self.classification_controller.do_classification_from_hand(
        #     pickle_file=chosen_pickle,
        #     test_subject=trained_subject,
        #     comparison_subject=,
        #     classifier_type=classifier_type,
        #     gesture_set=gesture_set,
        #     feature_set=feature_set,
        #     unseen_data=unseen_data
        # )

    def multiple_feature_classification(self):
        # Show files available for classification (pickle files)
        list_data_files = io.get_pickle_files()
        print("* List of Pickle Files *")
        printer.print_numbered_list(list_data_files)
        print("\n")

        # Prompt user for gesture name
        gesture_set, _, gesture_src = prompter.prompt_gesture_set()
        chosen_gesture = prompter.prompt_gesture_name(gesture_src)

        # Create time and classification dictionaries
        time_dict = {}
        cls_dict = {}

        # Create Leap Data Trainer for each feature type
        trainer_list = []
        for current_pickle in list_data_files:
            # Obtain pickle file name and kernel type
            current_pickle_no_extension = current_pickle.rsplit(".", 1)[0]
            kernel_type = current_pickle_no_extension.rsplit("_", 1)[-1]

            subject_name = self.classification_controller.subject_name
            # Create a Leap Data Trainer based on obtained pickle name and kernel type
            trainer = SVM_Trainer(subject_name=subject_name, feature_type=current_pickle_no_extension,
                                  kernel_type=kernel_type)
            trainer.load(current_pickle)

            # Append to list of trainers
            trainer_list.append(trainer)
            # Append to dictionaries
            time_dict[trainer] = []
            cls_dict[trainer] = 0

        # Do Classification TODO : Fix this for future tests
        # self.classification_controller.do_classification_from_hand(
        #     pickle_file=chosen_pickle,
        #     test_subject=trained_subject,
        #     comparison_subject=,
        #     classifier_type=classifier_type,
        #     gesture_set=gesture_set,
        #     feature_set=feature_set,
        #     unseen_data=unseen_data
        # )
    '''''
