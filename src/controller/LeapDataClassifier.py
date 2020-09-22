import time
from string import lower

from controller import LeapIO as io
from controller.LeapDataTrainer import NN_Trainer, SVM_Trainer, DT_Trainer
import controller.LeapFeatureExtractor as extractor

# Import view libraries
import view.prompt.Prompter as prompter


class LeapDataClassifier:
    def __init__(self, acquisitor):
        self.acquisitor = acquisitor

    def do_classification_from_csv(self, pickle_file, test_subject, comparison_subject, classifier_type, gesture_set,
                                   feature_set, unseen_data, file_name):
        # Obtain data content of the unseen data file
        X_data_set, y_data_set = io.acquire_data_from_csv(csv_file=unseen_data)
        trainer = None

        if lower(classifier_type) == 'nn':
            # Get set hyper parameters
            activation = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = NN_Trainer(subject_name=test_subject, feature_type=feature_set, activation=activation,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
            pass
        elif lower(classifier_type) == 'svm':
            # Get set hyper parameters
            kernel_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get SVM Trainer
            trainer = SVM_Trainer(subject_name=test_subject, feature_type=feature_set, kernel_type=kernel_type,
                                  gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
        elif lower(classifier_type) == 'dt':
            # Get set hyper parameters
            criterion_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = DT_Trainer(subject_name=test_subject, feature_type=feature_set, criterion_type=criterion_type,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)

        # Create time and classification lists
        time_list = []
        i = 0
        correct_predictions = 0
        # Do classification for each data set
        for X_data in X_data_set:
            y_data = y_data_set[i]

            # Classify gestures and return results
            result, time_taken, _ = self.classify_gesture(
                feature_data_set=X_data,
                feature_type=feature_set,
                chosen_gesture=y_data,
                trainer=trainer,
                verbose=True
            )

            time_list.append(time_taken)
            if result is True:
                correct_predictions += 1
                pass
            i += 1

        # Process corresponding results
        file_name = self.process_modified_test_results(
            classifier_type=classifier_type,
            test_subject=test_subject,
            correct_classification=correct_predictions,
            time_list=time_list,
            gesture_set=gesture_set,
            feature_set=feature_set,
            file_name=file_name,
            comparison_subject=comparison_subject,
            file_path=pickle_file,
            unseen_data=unseen_data,
            trainer=trainer,
            verbose=True
        )

        return file_name

    def do_classification_from_hand(self, pickle_file, test_subject, comparison_subject, classifier_type, gesture_set,
                                    feature_set, unseen_data, file_name):
        # Initialize variables
        file_name = None
        feature_data_set = None
        trainer = None

        # Obtain classifier type
        if lower(classifier_type) == 'nn':
            # Get set hyper parameters
            activation = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = NN_Trainer(subject_name=test_subject, feature_type=feature_set, activation=activation,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
            pass
        elif lower(classifier_type) == 'svm':
            # Get set hyper parameters
            kernel_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get SVM Trainer
            trainer = SVM_Trainer(subject_name=test_subject, feature_type=feature_set, kernel_type=kernel_type,
                                  gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
        elif lower(classifier_type) == 'dt':
            # Get set hyper parameters
            criterion_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = DT_Trainer(subject_name=test_subject, feature_type=feature_set, criterion_type=criterion_type,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)

        # Prompt for what gesture to be recorded
        gesture_set, _, gesture_src = prompter.prompt_gesture_set()
        chosen_gesture = prompter.prompt_gesture_name(gesture_src)

        # Acquire hand
        hand = self.acquisitor.acquire_single_hand_data()

        # Acquire X data set
        if feature_set == 'finger-angle-and-palm-distance':
            feature_name, feature_data_set = extractor.extract_finger_palm_distance(hand=hand)
            pass
        elif feature_set == 'finger-angle-using-bones':
            feature_name, feature_data_set = extractor.extract_finger_palm_angle(hand=hand)
            pass
        elif feature_set == 'finger-between-distance':
            feature_name, feature_data_set = extractor.extract_finger_palm_angle_distance(hand=hand)
            pass
        elif feature_set == 'finger-to-palm-distance':
            feature_name, feature_data_set = extractor.extract_finger_finger_distance(hand=hand)
            pass

        # Classify and return the result of the prediction
        self.classify_gesture(
            trainer=trainer,
            feature_type=feature_set,
            feature_data_set=feature_data_set,
            chosen_gesture=chosen_gesture,
        )

        print("")

    def classify_gesture(self, feature_data_set, feature_type, chosen_gesture, trainer, verbose=True):
        # Recording timing of classification
        start_time = time.time()
        prediction = trainer.classify([feature_data_set])
        end_time = time.time()
        time_taken = round(end_time - start_time, 5)

        # Output for user
        if (prediction[0]) == chosen_gesture:
            if verbose is True:
                print("- - - - - CORRECT PREDICTION - - - - -")
            result = True
        else:
            if verbose is True:
                print("+ + + + + INCORRECT PREDICTION + + + + +")
            result = False

        if verbose is True:
            print("Feature Used : " + feature_type)
            print("Prediction   : " + lower(prediction[0]))
            print("Time Taken   : " + str(time_taken) + "\n")

        return result, time_taken, prediction


    def process_modified_test_results(self, comparison_subject, test_subject, classifier_type, correct_classification,
                                      time_list, trainer,
                                      gesture_set, feature_set, file_name, file_path, unseen_data,
                                      verbose=False):
        # Calculate average time taken to perform classification algorithms between multiple test hand instances
        avg_time = (sum(time_list)) / (len(time_list))
        # Calculate average accuracy of classification algorithm between multiple test hand instances
        accuracy = round(100.0 * (float(correct_classification) / (float(len(time_list)))), 2)

        train_accuracy = round(trainer.training_acc * 100.0, 3)

        # Get pickle file name without folders
        pickle_file = file_path.split("\\")[-1].split(".")[0]
        unseen_data = unseen_data.split("\\")[-1].split(".")[0]
        if test_subject == comparison_subject:
            title = "PERSONALIZED TEST"
        else:
            title = "NON-PERSONALIZED TEST"

        summary = """

__________________________________________________________________________________________________
Test Subject Pickle    : %s 
Unseen Subject Data    : %s
__________________________________________________________________________________________________
        %s 
--------------------------------------------------------------------------------------------------
        Subject        :    %s
        Unseen Subject :    %s
        Feature        :    %s
        Gesture Set    :    %s
        Correct        :    %s
        Incorrect      :    %s
        Result         :    %s / %s
        Avg Time       :    %s seconds
        
        TRAINING 
        Accuracy       :    %s %%
        
        TESTING
        Accuracy       :    %s %%
        
        \n""" % (pickle_file,
                 unseen_data,
                 title,
                 test_subject,
                 comparison_subject,
                 feature_set,
                 gesture_set,
                 str(correct_classification),
                 str(len(time_list) - correct_classification),
                 str(correct_classification),
                 str(len(time_list)),
                 str(avg_time),
                 str(train_accuracy),
                 str(accuracy),
                 )

        # Print out results in summary form
        if verbose is True:
            print(summary)
            pass
        # Save summary onto report file
        return io.save_report(subject_name=test_subject, gesture_set=gesture_set, feature_set=feature_set,
                              report_header='classification', classifier_type=classifier_type, line=summary,
                              file_name=file_name)