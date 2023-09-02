import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras.models import load_model

test_datagen = ImageDataGenerator(rescale=1.0/255)
class_model = load_model('models/Food_mobnet.h5')

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\Hp\Downloads\Food_101\Food_101',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important: Do not shuffle the data to match with ground truth
)


predictions = class_model.predict(test_generator)

true_labels = test_generator.classes
predicted_labels = predictions.argmax(axis=-1)
confusion = confusion_matrix(true_labels, predicted_labels)

confusion_df = pd.DataFrame(confusion, columns=test_generator.class_indices.keys(), index=test_generator.class_indices.keys())

print(confusion_df)