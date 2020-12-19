import tensorflow as tf
import numpy as np
class model():
    model = tf.keras.models.load_model('model.h5')
    def predictions(self , image_array):
        image = image_array.reshape((1,28,28,1))/255.0
        p = self.model.predict(image)
        result = np.argmax(p)
        return result
