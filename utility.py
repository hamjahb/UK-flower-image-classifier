import tensorflow as tf
import tensorflow_hub as hub
import json

image_size = 224

def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)

        # remaping label indecies to match dataset
        class_names_new = dict()
        for key in class_names:
            class_names_new[str(int(key) - 1)] = class_names[key]
        return class_names_new


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model


def process_image(test_image):
    test_image = tf.convert_to_tensor(test_image, dtype=tf.float32)
    test_image = tf.image.resize(test_image, (image_size, image_size)).numpy()
    test_image /= 255
    return test_image