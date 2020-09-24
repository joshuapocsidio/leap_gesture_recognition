import controller.LeapIO as io
import time


class LeapHandAcquisitor:
    def __init__(self, leap_controller, supervised=False, verbose=False):
        self.leap_controller = leap_controller
        self.verbose = verbose
        self.supervised = supervised

        pass

    def acquire_single_hand_data(self):
        done = False
        loops = 0
        # Keep attempting until valid hand
        while done is False:
            # Only acquire if controller is connected
            if self.leap_controller.is_connected:
                hands = self.leap_controller.frame().hands
                # Only acquire if there are hands on screen
                if len(hands) > 0:
                    hand = hands[0]
                    return hand
                else:
                    print("\rWaiting for hand..."),
            else:
                print("\rLeap Motion Not Connected!"),
                return None

            loops += 1

    def acquire_multiple_hand_data(self, iterations=100, intervals=10):
        hand_set = []
        i = 0
        while i < iterations:
            time.sleep(0.05)
            # Acquire hand data
            hand = self.acquire_single_hand_data()
            if hand is not None:
                hand_set.append(hand)
                i += 1

            # If counter is at interval, prompt user
            if i % intervals == 0:
                if i == intervals:
                    raw_input("\nSystem       :       Gesture Checkpoint reached. Press any key to continue"),
                else:
                    raw_input("\nSystem       :       Press any key to get data: "),

        return hand_set
