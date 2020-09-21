import Leap

from controller.LeapDataClassifier import LeapDataClassifier
from controller.LeapHandAcquisitor import LeapHandAcquisitor
from view.menu import MainMenu

# Initialise Leap motion controller
controller = Leap.Controller()
listener = Leap.Listener()
controller.add_listener(listener)

def main():
    print("Gesture Application Booted")

    acquisitor = LeapHandAcquisitor(leap_controller=controller)
    classification_controller = LeapDataClassifier(acquisitor=acquisitor)
    MainMenu.show(acquisitor=acquisitor, classification_controller=classification_controller, leap_controller=controller)

if __name__ == '__main__':
    main()
