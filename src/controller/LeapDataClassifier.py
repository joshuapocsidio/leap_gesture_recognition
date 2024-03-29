import time
from string import lower, upper

from controller import LeapIO as io
from controller.LeapDataTrainer import NN_Trainer, SVM_Trainer, DT_Trainer
import controller.LeapFeatureExtractor as extractor

# Import view libraries
import view.prompt.Prompter as prompter


class LeapDataClassifier:
    def __init__(self, acquisitor):
        self.acquisitor = acquisitor

    def do_classification_from_csv(self, pickle_file, train_subject, test_subject, classifier_type, gesture_set,
                                   feature_set, unseen_data, file_name):
        # Obtain data content of the unseen data file
        X_data_set, y_data_set = io.acquire_data_from_csv(csv_file=unseen_data)
        trainer = None

        # Obtain classifier type
        trainer = self.obtain_classifier(classifier_type=classifier_type, gesture_set=gesture_set,
                                         feature_set=feature_set, train_subject=train_subject, pickle_file=pickle_file)

        # Create time and classification lists
        time_list = []
        i = 0
        correct_predictions = 0
        # Do classification for each data set
        print(""),
        for X_data in X_data_set:
            y_data = y_data_set[i]

            # Classify gestures and return results
            prediction, result, time_taken = self.classify_known_gesture(
                feature_data_set=X_data,
                feature_type=feature_set,
                chosen_gesture=y_data,
                trainer=trainer,
                verbose=False
            )

            time_list.append(time_taken)
            # Check if correct
            if result is True:
                correct_predictions += 1

            # Check if personalized
            if lower(train_subject) == lower(test_subject):
                personalized = "personalized"
            else:
                personalized = "non-personalized"

            print("\rProgress - - - > " + str(i) + "/" + str(len(X_data_set))),
            # Print regardless of verbosity
            # print(upper(personalized) + " : " + upper(classifier_type) +
            #       "(" + str(trainer.training_acc) + "%) -- Train Subject :" + test_subject +
            #       ", Test Subject : " + test_subject + " >> " + gesture_set + " - " + feature_set + " = " + prediction)

            # Append to csv results
            io.append_classification_csv_results(personalized=personalized, classifier_type=classifier_type,
                                                 training_score=trainer.training_acc, train_subject=test_subject,
                                                 test_subject=train_subject, gesture_set=gesture_set,
                                                 feature_set=feature_set, correct=result, time=time_taken,
                                                 gesture=y_data, prediction=prediction)
            i += 1

        # Process corresponding results
        self.process_modified_test_results(
            classifier_type=classifier_type,
            test_subject=train_subject,
            correct_classification=correct_predictions,
            time_list=time_list,
            gesture_set=gesture_set,
            feature_set=feature_set,
            file_name=file_name,
            comparison_subject=test_subject,
            file_path=pickle_file,
            unseen_data=unseen_data,
            trainer=trainer,
            verbose=False
        )

        accuracy = round(float(correct_predictions)/float(len(X_data_set)), 5)
        return accuracy

    def do_classification_from_features(self, trainer, feature_data_set):
        # Obtain values
        value_set = []
        for feature_data in feature_data_set:
            value_set.append(feature_data.value)

        prediction = self.classify_unknown_gesture(feature_data_set=value_set, trainer=trainer)

        return prediction[0]


    def do_classification_from_hand(self, pickle_file, train_subject, classifier_type, gesture_set,
                                    feature_set, chosen_gesture, hand):
        # Initialize variables
        feature_data_set = None

        # Obtain classifier type
        trainer = self.obtain_classifier(classifier_type=classifier_type, gesture_set=gesture_set,
                                         feature_set=feature_set, train_subject=train_subject, pickle_file=pickle_file)
        # Acquire X data set
        if feature_set == 'finger-angle-and-palm-distance':
            feature_name, feature_data_set = extractor.extract_finger_palm_angle_distance(hand=hand)
            pass
        elif feature_set == 'finger-angle-using-bones':
            feature_name, feature_data_set = extractor.extract_finger_palm_angle(hand=hand)
            pass
        elif feature_set == 'finger-between-distance':
            feature_name, feature_data_set = extractor.extract_finger_finger_distance(hand=hand)
            pass
        elif feature_set == 'finger-to-palm-distance':
            feature_name, feature_data_set = extractor.extract_finger_palm_distance(hand=hand)
            pass

        # Obtain just the values
        value_set = []
        for feature_data in feature_data_set:
            value_set.append(feature_data.value)

        prediction, result, _ = self.classify_known_gesture(
            trainer=trainer,
            feature_type=feature_set,
            feature_data_set=value_set,
            chosen_gesture=chosen_gesture,
            verbose=False
        )

        return prediction, result, trainer

    def classify_unknown_gesture(self, feature_data_set, trainer):
        prediction = trainer.classify([feature_data_set])
        return prediction


    def classify_known_gesture(self, feature_data_set, chosen_gesture, trainer, feature_type=None, verbose=True):
        # Recording timing of classification
        start_time = round(time.time(), 8)
        prediction = trainer.classify([feature_data_set])
        end_time = round(time.time(), 8)

        time_taken = round(end_time - start_time, 8)

        # Output for user
        if (prediction[0]) == chosen_gesture:
            if verbose is True:
                print("- - - - - CORRECT PREDICTION - - - - -")
            result = True
        else:
            if verbose is True:
                print("+ + + + + INCORRECT PREDICTION + + + + +")
            result = False

        if verbose is True and feature_type is not None:
            print("Feature Used : " + feature_type)
            print("Prediction   : " + lower(prediction[0]))
            print("Time Taken   : " + str(time_taken) + "\n")

        return prediction[0], result, time_taken

    def obtain_classifier(self, classifier_type, pickle_file, train_subject, feature_set, gesture_set):
        trainer = None
        # Obtain classifier type
        if lower(classifier_type) == 'nn':
            # Get set hyper parameters
            activation = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = NN_Trainer(subject_name=train_subject, feature_type=feature_set, activation=activation,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
            pass
        elif lower(classifier_type) == 'svm':
            # Get set hyper parameters
            kernel_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get SVM Trainer
            trainer = SVM_Trainer(subject_name=train_subject, feature_type=feature_set, kernel_type=kernel_type,
                                  gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)
        elif lower(classifier_type) == 'dt':
            # Get set hyper parameters
            criterion_type = pickle_file.split(".")[0].split("--")[1].split("_")[1]
            # Get NN Trainer
            trainer = DT_Trainer(subject_name=train_subject, feature_type=feature_set, criterion_type=criterion_type,
                                 gesture_set=gesture_set)
            trainer.load(pickle_name=pickle_file)

        return trainer

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
