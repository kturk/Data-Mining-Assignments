from PIL import Image
import numpy as np
from numpy import asarray
import os
import operator

'''
Returns the given input image as vector.
'''
def image_to_vector(input_image):

    image = Image.open(input_image)                                 # Opening the image.
    input_image_as_array  = asarray(image)                          # Converting image to an array.
    input_image_as_array  = input_image_as_array.astype('float32')  # Changing the array's elements to 'float32' type.
    input_image_as_vector = input_image_as_array.flatten()          # Flatten the array

    return input_image_as_vector

'''
Returns a dictionary of all image names and their similarity with the given 'input_vector'.
'''
def cosine_similarity(input_vector):

    cos_sim_dictionary = {}                       # Dictionary to keep key-value pairs of image name and similarity.
    directory          = os.fsencode('Car_Data')  # Having the directory path

    for file in os.listdir(directory):  # Iterating through all files in 'Car_Data'
        filename        = os.fsdecode(file)           # Having the current file name.
        file_path       = 'Car_Data/' + filename      # Adding file name 'Car_Data/' to have file path.
        image_as_vector = image_to_vector(file_path)  # Converting the image into vector.

        dot_product          = np.dot(input_vector, image_as_vector)  # Dot product of input vector and current image(vector).
        input_vector_norm    = np.linalg.norm(input_vector, ord=2)
        image_as_vector_norm = np.linalg.norm(image_as_vector, ord=2)
        cos_sim              = dot_product / (input_vector_norm * image_as_vector_norm)  # Cosine similarity of two vectors

        cos_sim_dictionary[filename] = cos_sim  # Adding the the current file and similarity to the dictionary.

    cos_sim_dictionary = dict(sorted(cos_sim_dictionary.items(), key=operator.itemgetter(1)))  # Sorting the dictionary to have
                                                                                               # the biggest similarities at the
                                                                                               # beginning of the dictionary.
    return cos_sim_dictionary


if __name__ == '__main__':

    print("Most similar ones to 4228.png:")
    inputImageAsVector = image_to_vector('Car_Data/4228.png')
    dictionary1        = cosine_similarity(inputImageAsVector)
    dictionary1.popitem()  # To get rid of 1.0 similarity (Same picture of '4228.png')
    for i in range(3):
        key, value = dictionary1.popitem()
        print('key: ', key, ' value: ', value)

    print("----------------------------------------")

    print("Most similar ones to 3861.png:")
    inputImageAsVector = image_to_vector('Car_Data/3861.png')
    dictionary2        = cosine_similarity(inputImageAsVector)
    dictionary2.popitem()  # To get rid of 1.0 similarity (Same picture of '3861.png')
    for i in range(3):
        key, value = dictionary2.popitem()
        print('key: ', key, ' value: ', value)