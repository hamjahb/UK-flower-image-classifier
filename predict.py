import argparse
import numpy as np
import json

import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from utility import process_image, get_class_names, load_model


def predict(image_path, model_path, top_k, class_names_new):
    top_k = int(top_k)
    model = load_model(model_path)
    image = Image.open(image_path)

    # image pre-precessing
    test_image = np.asarray(image)
    processed_test_image = process_image(test_image)

    # probablitites
    prob_preds = model.predict(np.expand_dims(processed_test_image,axis=0))
    prob_preds = prob_preds[0].tolist()

    values, classes= tf.math.top_k(prob_preds, k=top_k)
    
    probs = values.numpy().tolist()
    classes = classes.numpy().tolist()
    
    print("prediction probabilities :\n", probs)
    print('prediction classes:\n', classes)
    
    
    for predicted_class in classes:
        print(class_names_new[str(predicted_class)])
    
    return probs, classes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "UK flower image classifier")
    parser.add_argument("image_path", help="Image Path")
    parser.add_argument("saved_model", help="Model Path")
    parser.add_argument("--top_k", help="Fetch top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()

    all_class_names = get_class_names(args.category_names)


    predict(args.image_path, args.saved_model, args.top_k, all_class_names)


""" Example commands to run in command line 
    
    python predict.py ./test_images/wild_pansy.jpg Trained_Model/Model.h5 
    python predict.py ./test_images/wild_pansy.jpg Trained_Model/Model.h5 --top_k 5
    python predict.py ./test_images/wild_pansy.jpg Trained_Model/Model.h5 --category_names label_map.json


"""