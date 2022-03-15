import tensorflow as tf
import keras2onnx

def load_model(json_path, weight_path):
    with open(json_path, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json, custom_objects={})
    model.load_weights(weight_path)
    return model

weights = "/home/ngoc/work/ai_acd/LPD/Detect_LP/wpod-net_update1.h5"
model_json = "/home/ngoc/work/ai_acd/LPD/Detect_LP/wpod-net_update1.json"
model = load_model(model_json, weights)
onnx_model = keras2onnx.convert_keras(model, name="wpod")
keras2onnx.save_model(onnx_model, "wpod.onnx")