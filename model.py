from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_age_and_gender(image_var):
    # Load the trained model
    model1 = load_model("./model/age_prediction_model_v4.h5")
    model2 = load_model("./model/age_and_gender_prediction_model_v3.h5")
    
    # Preprocess the image
    img_array = img_to_array(image_var) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model2.predict(img_array)[0]

    predicted_age = model1.predict(img_array)[0][0]

    predicted_gender = "Male" if predictions[1] < 0.5 else "Female"  # Assuming the second output neuron represents gender (0: Male, 1: Female)
    print(predicted_age, predicted_gender)
    return int(predicted_age)+1, predicted_gender

# Example usage:
# Assuming you have an image variable named 'image_var'
# predicted_age, predicted_gender = predict_age_and_gender('D.jpg')
# print("Predicted age:", predicted_age)

