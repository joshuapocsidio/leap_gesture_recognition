import AcquisitionMenu, RecognitionMenu, TrainingMenu, ClassificationMenu
from ClassificationMenu import ClassificationMenu

def show(acquisitor, classification_controller, leap_controller):
    # Shows menu for main menu
    done = False

    while done is False:
        print("---------")
        print("MAIN MENU")
        print("---------")
        print("(1) - Gesture Data Acquisition")
        print("(2) - Gesture Data Training")
        print("(3) - Gesture Classification Testing")
        print("(4) - Real Time Gesture Recognition")
        print("(0) - Exit Program")

        choice = raw_input("Your Choice: ")
        print("YOUR INPUT: " + choice)
        if choice == '1':
            AcquisitionMenu.show(acquisitor=acquisitor)
        elif choice == '2':
            TrainingMenu.show()
            pass
        elif choice == '3':
            classification_menu = ClassificationMenu(classification_controller=classification_controller)
            classification_menu.show()
            pass
        elif choice == '4':
            recognition_menu = RecognitionMenu.show(leap_controller)
            pass
        elif choice == '0':
            # Shows GUI for exiting the program
            done = True