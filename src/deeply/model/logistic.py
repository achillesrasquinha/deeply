from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def LogisticRegression(
    backbone = None
):
    model = Sequential()
    # model.add(Input())

    model.add(Dense(2, activation = "softmax"))

    model.compile(
        optimizer = "sgd",
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )

    return model