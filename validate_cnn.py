"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
from data import DataSet
from processor import process_image
from keras.models import load_model
from collections import Counter

def main():
    """Spot-check `nb_images` images."""
    #data = DataSet()
    classes = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
    model = load_model('/home/ubuntu/workspace/emai/data/checkpoints/inception-007-0.43.hdf5')
    #model = load_model('./data/checkpoints/inception.039-1.69.hdf5')
    #model = load_model('./data/checkpoints/inception.002-1.95.hdf5')
    likelihood = []
    probability = []
    # Get all our test images.
    images = glob.glob('/home/ubuntu/workspace/emai/data/validation_data/*.jpg')
    for image in images:
    #for _ in range(nb_images):
        #print('-'*80)
        # Get a random row.
        #sample = random.randint(0, len(images) - 1)
        #image = images[sample]

        # Turn the image into an array.
        #print(image)
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)
        #import pdb; pdb.set_trace()
        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            #import pdb;pdb.set_trace()
            if i >= 1:
                break
            likelihood.append(class_prediction[0])
            probability.append((class_prediction[0],class_prediction[1]))
            #print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            #i += 1
    print(Counter(likelihood))
    return Counter(likelihood)
    #import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
