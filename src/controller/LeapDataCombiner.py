# Controller libraries
import controller.LeapIO as io

# Built in libraries
from string import strip


def combine_gestures_separate_subjects():
    # Get parameters
    data_files, _, unseen_data_files, _, _, feature_set_list, _, subject_name_list = io.get_params()

    # Grouping Stage
    group_by_subject = []
    unseen_group_by_subject = []
    i = 0

    # Group by Subject Name
    for subject_name in subject_name_list:
        subject_group = []
        unseen_subject_group = []
        j = 0

        # Group by Feature Set
        for feature_set in feature_set_list:
            # raw_input(feature_set + " -- " + subject_name)
            # Acquired Data
            subject_feature_group = []
            for data_file in data_files:
                feature_set_check = data_file.split("--")[1].split(".")[0]
                subject_name_check = data_file.split("(")[1].split(")")[0]

                # Only append to group if matches subject and feature
                if feature_set == feature_set_check and subject_name == subject_name_check:
                    subject_feature_group.append(data_file)

            if len(subject_feature_group) > 0:
                subject_group.append(subject_feature_group)

            # Unseen Data
            unseen_subject_feature_group = []
            for unseen_data_file in unseen_data_files:
                feature_set_check = unseen_data_file.split("--")[1].split(".")[0]
                subject_name_check = unseen_data_file.split("(")[1].split(")")[0]

                # Only append to group if matches subject and feature
                if feature_set == feature_set_check and subject_name == subject_name_check:
                    unseen_subject_feature_group.append(unseen_data_file)

            if len(unseen_subject_feature_group) > 0:
                unseen_subject_group.append(unseen_subject_feature_group)

            j += 1

        # Only Append to group if not an empty group
        if len(subject_group) > 0:
            group_by_subject.append(subject_group)

        # Unseen Data
        if len(unseen_subject_group) > 0:
            unseen_group_by_subject.append(unseen_subject_group)
        i += 1

    # Creating data file for combined acquired data
    for group in group_by_subject:
        for file_item in group:
            # All items in this group should have same subject and same content
            subject = file_item[0].split("(")[1].split(")")[0]
            feature_set = file_item[0].split("--")[1].split(".")[0]

            # Construct file name
            file_name = io.com_dir + "(" + subject + ") " + "COMBINED GESTURES--" + feature_set + ".csv"

            # Create the file
            file_creation(group=file_item, file_name=file_name)

    # Creating data file for combined unseen data
    for group in unseen_group_by_subject:
        for file_item in group:
            # All items in this group should have same subject and same content
            subject = file_item[0].split("(")[1].split(")")[0]
            feature_set = file_item[0].split("--")[1].split(".")[0]

            # Construct file name
            file_name = io.cun_dir + "(" + subject + ") " + "COMBINED GESTURES--" + feature_set + ".csv"

            # Create the file
            file_creation(group=file_item, file_name=file_name)
    pass


def combine_subjects_separate_gestures():
    # Get parameters
    data_files, _, unseen_data_files, _, _, feature_set_list, gesture_set_list, _ = io.get_params()

    # Grouping Stage
    group_by_gesture = []
    unseen_group_by_gesture = []
    i = 0

    # Group by Gesture SetName
    for gesture_set in gesture_set_list:
        gesture_group = []
        unseen_gesture_group = []
        j = 0

        # Group by Feature Set
        for feature_set in feature_set_list:
            # Acquired Data
            gesture_feature_group = []
            for data_file in data_files:
                feature_set_check = data_file.split("--")[1].split(".")[0]
                gesture_set_check = strip(data_file.split("--")[0].split(")")[1])

                # Only append to group if matches subject and feature
                if feature_set == feature_set_check and gesture_set == gesture_set_check:
                    gesture_feature_group.append(data_file)

            if len(gesture_feature_group) > 0:
                gesture_group.append(gesture_feature_group)

            # Unseen Data
            unseen_gesture_feature_group = []
            for unseen_data_file in unseen_data_files:
                feature_set_check = unseen_data_file.split("--")[1].split(".")[0]

                # Only append to group if matches subject and feature
                if feature_set == feature_set_check:
                    unseen_gesture_feature_group.append(unseen_data_file)

            if len(unseen_gesture_feature_group) > 0:
                unseen_gesture_group.append(gesture_feature_group)
            j += 1

        # Acquired Data
        if len(gesture_group) > 0:
            group_by_gesture.append(gesture_group)

        # Unseen Data
        if len(unseen_gesture_group) > 0:
            unseen_group_by_gesture.append(unseen_gesture_group)

        i += 1

    # Creating data file for combined acquired data
    for group in group_by_gesture:
        for file_item in group:
            # All items in this group should have same subject and same content
            gesture_set = strip(file_item[0].split("--")[0].split(")")[1])
            feature_set = file_item[0].split("--")[1].split(".")[0]

            # Construct file name
            file_name = io.com_dir + "(COMBINED SUBJECTS) " + gesture_set + "--" + feature_set + ".csv"

            # Create the file
            file_creation(group=file_item, file_name=file_name)

    # Creating data file for combined unseen data
    for group in unseen_group_by_gesture:
        for file_item in group:
            # All items in this group should have same subject and same content
            gesture_set = strip(file_item[0].split("--")[0].split(")")[1])
            feature_set = file_item[0].split("--")[1].split(".")[0]

            # Construct file name
            file_name = io.cun_dir + "(COMBINED SUBJECTS) " + gesture_set + "--" + feature_set + ".csv"

            # Create the file
            file_creation(group=file_item, file_name=file_name)

    pass


def combine_subjects_combine_gestures():
    # Get parameters
    data_files, _, unseen_data_files, _, _, feature_set_list, _, _ = io.get_params()

    # Grouping Stage
    combined_acquired_data = []
    combined_unseen_data = []
    i = 0

    # Group by Feature Set
    for feature_set in feature_set_list:
        # Acquired Data
        for data_file in data_files:
            feature_set_check = data_file.split("--")[1].split(".")[0]

            # Only append to group if matches subject and feature
            if feature_set == feature_set_check:
                combined_acquired_data.append(data_file)

        # Unseen Data
        for unseen_data_file in unseen_data_files:
            feature_set_check = unseen_data_file.split("--")[1].split(".")[0]

            # Only append to group if matches subject and feature
            if feature_set == feature_set_check:
                combined_unseen_data.append(unseen_data_file)

        i += 1

    # Creating data file for combined acquired data
    for file_item in combined_acquired_data:
        # All items in this group should have same subject and same content
        feature_set = file_item.split("--")[1].split(".")[0]

        # Construct file name
        file_name = io.com_dir + "(COMBINED SUBJECTS) " + "COMBINED GESTURES--" + feature_set + ".csv"

        # Create the file
        file_creation(single_item=file_item, file_name=file_name, single=True)

    # Creating data file for combined unseen data
    for file_item in combined_unseen_data:
        # All items in this group should have same subject and same content
        feature_set = file_item.split("--")[1].split(".")[0]

        # Construct file name
        file_name = io.cun_dir + "(COMBINED SUBJECTS) " + "COMBINED GESTURES--" + feature_set + ".csv"

        # Create the file
        file_creation(single_item=file_item, file_name=file_name, single=True)

    pass


def file_creation(file_name, group=None, single_item=None, single=False):
    if single is True and single_item is not None:
        # Get the content and labels from the file
        content = io.read_all(single_item)
        labels = strip(str(content[0])).split(",")
        del content[0]
        content = "".join(content)

        if io.does_file_exist(file_name=file_name) is False:
            io.create_data_file(file_name=file_name, labels=labels)

        io.append_to_file(file_name=file_name, lines=strip(str(content)))
    else:
        for file_item in group:
            # Get the content and labels from the file
            content = io.read_all(file_item)
            labels = strip(str(content[0])).split(",")
            del content[0]
            content = "".join(content)
            if io.does_file_exist(file_name=file_name) is False:
                io.create_data_file(file_name=file_name, labels=labels)

            io.append_to_file(file_name=file_name, lines=strip(str(content)))

