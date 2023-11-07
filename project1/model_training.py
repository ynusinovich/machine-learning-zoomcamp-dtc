import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
import xgboost as xgb

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

df_train = pd.read_csv('df_train.csv')
df_val = pd.read_csv('df_val.csv')
df_test = pd.read_csv('df_test.csv')

dv = DictVectorizer(sparse=True)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]

model_df = pd.DataFrame(columns = ["counter", "eta", "max_depth", "min_child_weight", "subsample", "f1", "threshold"])
counter = 0
models = []
for eta in (0.1, 0.3, 0.5):
    for max_depth in (4, 6, 8):
        for min_child_weight in (1, 2, 3):
            for subsample in (0.5, 1):

                xgb_params = {
                              'eta': eta, 
                              'max_depth': max_depth,
                              'min_child_weight': min_child_weight,
                              'subsample': subsample,
                              'objective': 'binary:logistic',                    
                              'verbosity': 1,
                             }

                model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                                verbose_eval=5,
                                evals=watchlist)
                
                y_pred_proba = model.predict(dval)

                final_threshold = 0
                final_f1 = 0
                for threshold in np.arange(0, 1.001, 1/1000):
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    if f1_score(y_val, y_pred) > final_f1:
                        final_threshold = threshold
                        final_f1 = f1_score(y_val, y_pred)
                
                new_data = {
                            "counter": counter,
                            "eta": xgb_params["eta"],
                            "max_depth": xgb_params["max_depth"],
                            "min_child_weight": xgb_params["min_child_weight"],
                            "subsample": xgb_params["subsample"],
                            "f1": final_f1,
                            "threshold": final_threshold
                           }

                new_row = pd.DataFrame(new_data, index = [0])

                model_df = pd.concat([model_df, new_row], ignore_index=True)

                counter += 1
                models.append(model)

model_df.to_csv("model_training_results.csv", index=False)

with open('dv.bin', 'wb') as f:
    pickle.dump(dv, f)

best_model_df = model_df.sort_values(by="f1", ascending=False).iloc[0]
best_model_counter = best_model_df["counter"]
best_model_f1 = best_model_df["f1"]
best_model_threshold = best_model_df["threshold"]
best_model = models[best_model_counter]
best_model.save_model('best_model.model')

print(f"final f1: {best_model_f1}")
print(f"final threshold: {best_model_threshold}")