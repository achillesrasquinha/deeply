from deeply.model.mlp import MLPRegressor

def LinearRegression(*args, **kwargs):
    kwargs["activation"]   = None
    kwargs["hidden"]  = []
    kwargs["learning_rate"] = kwargs.get("learning_rate", 0.1)
    kwargs["name"] = kwargs.get("name", "linreg")

    return MLPRegressor(*args, **kwargs)