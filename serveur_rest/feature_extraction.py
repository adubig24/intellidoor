from keras.preprocessing import image
from keras_facenet import FaceNet
import cv2
import numpy as np
import pandas as pd

class feature_extraction:
    def model(self):
        model = FaceNet()
        return model  # return the model here
        
    def feature_extraction(self, img):
        model = self.model()
        img = np.expand_dims(img, axis=0)
        feature = model.embeddings(img)
        return feature
    
    def transformation_dataframe(self,feature):
        dataframe = pd.DataFrame(np.array(feature).reshape(-1,len(feature)))
        dataframe = dataframe.T
        return dataframe
    
    def sauvegarde_csv(self,dataframe, img):
        dataframe.to_csv("../data/data.csv", index=False)
        cv2.imwrite("../data/face.jpg", img)
        return dataframe