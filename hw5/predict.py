import pickle
import argparse
import json
import os


def predict(customer):

    with open("dv.bin", 'rb') as f_in:
        dv = pickle.load(f_in)

    if os.path.isfile("model1.bin"):
        with open("model1.bin", 'rb') as f_in:
            model = pickle.load(f_in)

    elif os.path.isfile("model2.bin"):
        with open("model2.bin", 'rb') as f_in:
            model = pickle.load(f_in)

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'probability': float(y_pred)
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process customer data from the command line.")
    parser.add_argument("--data", type=str, help="JSON data representing customer information")

    args = parser.parse_args()

    if args.data:
        try:
            customer = json.loads(args.data)
            print("Customer data received:")
            print(customer)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON data. {e}")
    else:
        print("No data provided. Use the --data flag to provide customer information.")
    
    result = predict(customer)
    print(result)