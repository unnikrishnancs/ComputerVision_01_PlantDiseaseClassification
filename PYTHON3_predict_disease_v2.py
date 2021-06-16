from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

def load_image(img_path, show=False):

    #img = image.load_img(img_path, target_size=(150, 150))
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)                    # (height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_array /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_array[0])                           
        plt.axis('off')
        plt.show()

    return img_array

if __name__ == "__main__":

    # load model
    #model = load_model("/home/ai16/Desktop/data/ai16_Data/cnn_model.h5")
    model = load_model("/home/ai16/project/main_project/cnn_model.h5")

    # image path
    img_path = '/home/ai16/project/main_project/Data/TEST/Pepper_bell_Bacterial_spot/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/Pepper_bell_healthy/.JPG' 
    img_path = '/home/ai16/project/main_project/Data/TEST/Potato_Early_blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/Potato_healthy/0b3e5032-8ae8-49ac-8157-a1cac3df01dd___RS_HL 1817.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'
    img_path = '/home/ai16/project/main_project/Data/TEST/.JPG'


    # load a single image
    new_image = load_image(img_path)

    new_list = ['Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy', 'Potato_Early_blight',\
 'Potato_Late_blight', 'Potato_healthy', 'Tomato_Bacterial_spot',\
 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',\
 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites', 'Tomato_Target_Spot',\
 'Tomato_YellowLeaf_Curl_Virus', 'Tomato_healthy', 'Tomato_mosaic_virus']

	

    # check prediction
    #Generates output predictions for the input samples.
    #Computation is done in batches
    print("Predicted Probabilities.......")
    pred = model.predict(new_image)
    print(pred)
    print("Maximum Probaility....."+ str(pred.max()))
     
    print("Index with highest probability .....")
    #Index with largest value across axes of tensor..
    classes_names=pred.argmax(axis=-1)
    print(classes_names)
    print(new_list[classes_names[0]])
    #classes_names_1=pred.argmax()
    #print pred[classes_names]

    img_vw=cv.imread(img_path)
    cv.imshow('Input Image',img_vw)
    #cv.waitKey(0) 
    cv.waitKey(10000) 
    #cv.DestroyWindow("Input Image")


"""
    print "retrieve most likely result, e.g highest probability"
    label=label[0][0]
    print label

print "classification"
print('%s (%.2f%%)'%(label[1],label[2]*100))



	print("Prediction",np.argmax(Ytr[nn_index]))
	print("True Class",np.argmax(Yte[i]))
"""






"""
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# dimensions of our images
img_width, img_height = 320, 240

# load the model we saved
model = load_model('model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
img = image.load_img('test1.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print classes

# predicting multiple images at once
img = image.load_img('test2.jpg', target_size=(img_width, img_height))
y = image.img_to_array(img)
y = np.expand_dims(y, axis=0)

# pass the list of multiple images np.vstack()
images = np.vstack([x, y])
classes = model.predict_classes(images, batch_size=10)

# print the classes, the images belong to
print classes
print classes[0]
print classes[0][0]
"""




