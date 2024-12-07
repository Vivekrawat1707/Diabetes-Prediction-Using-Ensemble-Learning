import sys
from main import main
from predict import main as predict_main

def print_menu():
    print("\nDiabetes Prediction System")
    print("=" * 30)
    print("1. Train and evaluate models")
    print("2. Make new predictions")
    print("3. Exit")

def run():
    while True:
        print_menu()
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            main()
        elif choice == '2':
            predict_main()
        elif choice == '3':
            print("\nThank you for using the Diabetes Prediction System!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    run()