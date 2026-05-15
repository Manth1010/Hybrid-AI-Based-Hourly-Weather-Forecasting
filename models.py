from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor

def get_lightgbm():
    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=31
        )
    )

def get_catboost():
    return MultiOutputRegressor(
        CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.03,
            verbose=0
        )
    )