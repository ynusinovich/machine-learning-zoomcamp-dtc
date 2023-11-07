import pickle
import argparse
import json
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def smiles_to_fp_array(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            radius = 2
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius)
            fp_array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            return fp_array
        else:
            print(f"No molecule for SMILES {smiles}")
            return np.zeros((2048,))
    except Exception as e:
        print(e)
        print(f"Error for SMILES {smiles}")


def predict(molecule):

    # load threshold for maximum f-1
    if os.path.isfile("model_training_results.csv"):
        model_df = pd.read_csv("model_training_results.csv")
        best_model_df = model_df.sort_values(by="f1", ascending=False).iloc[0]
        best_threshold = best_model_df["threshold"]
    else:
        best_threshold = 0.149

    # load dictionary vectorizer
    with open("dv.bin", 'rb') as f_in:
        dv = pickle.load(f_in)

    # load XGBoost model
    model = xgb.Booster()
    model.load_model('best_model.model')

    # get feature names
    features = list(dv.get_feature_names_out())

    # create dataframe for fingerprint of molecule
    fp = smiles_to_fp_array(molecule["molecule"])
    fp_df = pd.DataFrame([fp], columns=[str(i) for i in range(2048)])
    fp_dict = fp_df.to_dict(orient='records')

    # transform fingerprint into vectorized dictionary and then DMatrix
    X = dv.transform(fp_dict)
    dpred = xgb.DMatrix(X, feature_names=features)
    
    # predict probabilities
    y_pred_proba = model.predict(dpred)

    # get prediction with threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    if y_pred == 1:
        is_toxic = "The molecule is toxic."
    else:
        is_toxic = "The molecule is not toxic."
    result = {
        'toxic': is_toxic
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process molecule data from the command line.")
    parser.add_argument("--data", type=str, help="JSON data representing molecule information")

    args = parser.parse_args()

    if args.data:
        try:
            molecule = json.loads(args.data)
            print("Molecule data received:")
            print(molecule)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON data. {e}")
    else:
        print("No data provided. Use the --data flag to provide customer information.")
    
    result = predict(molecule)
    print(result)