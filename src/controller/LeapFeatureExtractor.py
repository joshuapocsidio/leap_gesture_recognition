from model.LeapData import LeapData
import controller.LeapIO as io
import math

from Leap import Bone, Finger, Vector


def extract_finger_palm_distance(hand):
    feature_name = 'finger-to-palm-distance'
    labels = ["thumb", "index", "middle", "ring", "pinky", "palm_normal_angle"]

    # Obtain fingers
    fingers = hand.fingers

    # Obtain palm vector position
    palm_vector = hand.palm_position

    # Iterate through each finger and obtain distance data
    value_set = []
    for finger in fingers:
        distance = round(finger.tip_position.distance_to(palm_vector), 5)
        # Add the hand data to data set
        value_set.append(distance)

    # Append palm normal direction
    value_set.append(extract_palm_normal_angle_to_y_direction(hand=hand))
    # Obtain LeapData set
    feature_data_set = combine_labels_value(labels=labels, values=value_set)

    return feature_name, feature_data_set


def extract_finger_palm_angle(hand):
    feature_name = 'finger-angle-using-bones'
    labels = ["thumb", "index", "middle", "ring", "pinky", "palm_normal_angle"]

    # Obtain fingers
    fingers = hand.fingers

    # Iterate through each finger and execute appropriate adjustments
    value_set = []
    for finger in fingers:
        if finger.type == Finger.TYPE_THUMB:
            # Use Proximal and Distal bones and obtain direction vectors
            proximal_bone = finger.bone(Bone.TYPE_PROXIMAL)
            distal_bone = finger.bone(Bone.TYPE_DISTAL)

            inner_bone_vector = proximal_bone.direction
            outer_bone_vector = distal_bone.direction
        else:
            # Use Metacarpal and Intermediate bones and obtain direction vectors
            metacarpal_bone = finger.bone(Bone.TYPE_METACARPAL)
            intermediate_bone = finger.bone(Bone.TYPE_INTERMEDIATE)

            inner_bone_vector = metacarpal_bone.direction
            outer_bone_vector = intermediate_bone.direction

        # Calculate angle between inner and outer bones' direction vectors
        angle_rad = inner_bone_vector.angle_to(outer_bone_vector)
        angle_deg = math.degrees(angle_rad)

        # Add the data to data_set
        value_set.append(angle_deg)

    # Append palm normal direction
    value_set.append(extract_palm_normal_angle_to_y_direction(hand=hand))
    # Obtain LeapData set
    feature_data_set = combine_labels_value(labels=labels, values=value_set)

    return feature_name, feature_data_set


def extract_finger_palm_angle_distance(hand):
    feature_name = 'finger-angle-and-palm-distance'
    labels = ["thumb", "index", "middle", "ring", "pinky", "palm_normal_angle"]

    # Obtain fingers
    fingers = hand.fingers

    # Iterate through each finger and obtain angle and distance data
    value_set = []

    # Obtain both information
    _, angle_set = extract_finger_palm_angle(hand=hand)
    _, distance_set = extract_finger_palm_distance(hand=hand)

    # Number of distances and angle should be the same
    num_angles = len(angle_set)
    num_distances = len(distance_set)

    if num_angles == num_distances:
        # Iterate through each data - up to number of data obtained
        i = 0
        while i < num_angles and i < num_distances:
            angle = float(angle_set[i].value)
            distance = float(distance_set[i].value)

            # Combine the two data by multiplying
            value = round(angle * distance, 5)
            # Append value to data set
            value_set.append(value)

            i += 1
        pass

    # Append palm normal direction
    value_set.append(extract_palm_normal_angle_to_y_direction(hand=hand))
    # Obtain LeapData set
    feature_data_set = combine_labels_value(labels=labels, values=value_set)

    return feature_name, feature_data_set


def extract_finger_finger_distance(hand):
    feature_name = 'finger-between-distance'
    labels = ["thumb-index", "index-middle", "middle-ring", "ring-pinky", "palm_normal_angle"]

    # Obtain fingers
    fingers = hand.fingers

    # Iterate through each finger and execute appropriate adjustments
    value_set = []

    # For each hand, iterate through each finger pairs and obtain data
    i = 0
    while i < (len(fingers) - 1):
        # Get fingers
        finger_a = fingers[i]
        finger_b = fingers[i + 1]

        # Get finger tip positions
        vector_a = finger_a.tip_position
        vector_b = finger_b.tip_position

        # Get distance between finger a and b
        distance = round(vector_a.distance_to(vector_b), 5)

        # Add the data to data_set
        value_set.append(distance)

        i += 1

    # Append palm normal direction
    value_set.append(extract_palm_normal_angle_to_y_direction(hand=hand))
    # Obtain LeapData set
    feature_data_set = combine_labels_value(labels=labels, values=value_set)

    return feature_name, feature_data_set


def extract_all_feature_type(hand):
    feature_map = []

    feature_map.append(extract_finger_palm_distance(hand=hand))
    feature_map.append(extract_finger_palm_angle(hand=hand))
    feature_map.append(extract_finger_palm_angle_distance(hand=hand))
    feature_map.append(extract_finger_finger_distance(hand=hand))

    return feature_map


def extract_palm_normal_angle_to_y_direction(hand):
    # Vector corresponding to UP direction
    up_vec = Vector.up
    # Obtain palm normal
    palm_normal = hand.palm_normal
    # Calculate angle between palm normal vector and UP vector - converting to degrees
    angle_deg = math.degrees(palm_normal.angle_to(up_vec))

    return angle_deg


def extract_palm_basis(hand):
    if hand is not None:
        x_d = round(hand.palm_normal.x, 5)
        y_d = round(hand.palm_normal.y, 5)
        z_d = round(hand.palm_normal.z, 5)

        return x_d, y_d, z_d


def combine_labels_value(labels, values):
    # Iterate through all values obtained and convert to Leap Data
    i = 0
    data_set = []
    for value in values:
        data = LeapData(label=labels[i], value=value)
        data_set.append(data)
        i += 1

    return data_set
