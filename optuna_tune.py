import optuna
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

def tune_lgbm(X, y):

    def objective(trial):
        model = MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2)
            )
        )

        model.fit(X, y)
        return model.score(X, y)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    return study.best_params