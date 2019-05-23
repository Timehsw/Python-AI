import xgboost as xgb

import joblib
#save model
# joblib.dump(xgb, filename)

path='/Users/hushiwei/Downloads/归档/dt+th+gd/model/xgb.ml'
#load saved model
obj = joblib.load(path)

# obj=xgb.Booster(model_file=path)
#
print(obj)
#
# print(path)