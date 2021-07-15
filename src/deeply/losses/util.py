from tensorflow.python.framework.ops import (
    convert_to_tensor_v2_with_dispatch
)

def preprocess_y(y_true, y_pred):
    y_true = convert_to_tensor_v2_with_dispatch(y_true)
    y_pred = convert_to_tensor_v2_with_dispatch(y_pred)
    
    # shape  = y_true.shape
    # shape.assert_is_compatible_with(y_pred.shape)

    return y_true, y_pred