import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def predict(use_model="vgg16", image_path=r"images\137.jpg"):
    MODELS = {
        "vgg16": VGG16,
        "vgg19": VGG19,
        "inception": InceptionV3,
        "xception": Xception,
        "resnet": ResNet50
    }

    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input

    if use_model in ["inception", "xception"]:
        inputShape = (299, 299)
        preprocess = preprocess_input

    print(f"[+] loading {use_model} ... ", end="")
    selected_network = MODELS[use_model]
    model = selected_network(weights="imagenet")
    print("Done")

    print(f"[#] processing {image_path} ... ", end="")
    image = load_img(image_path, target_size=inputShape)
    image = img_to_array(image)
    print("Done")

    image = np.expand_dims(image, axis=0)
    image = preprocess(image)

    print(f"[#] classifying image with '{use_model}'... ", end="")
    predictions = model.predict(image)
    top_predictions = imagenet_utils.decode_predictions(predictions)
    print("Done")

    print("\n[-] Top five predictions are:")
    for (i, (imagenetID, label, prob)) in enumerate(top_predictions[0]):
        print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

    # orig = Image.open(image_path)
    (imagenetID, label, prob) = top_predictions[0][0]
    # plt.title("{}, {:.2f}%".format(label, prob * 100))
    # plt.imshow(orig)
    # plt.show()

    return label, prob
