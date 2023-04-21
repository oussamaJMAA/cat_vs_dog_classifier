import streamlit as st
from keras.models import load_model
import keras.utils as image
import numpy as np

# Load the trained model
model = load_model('cat_vs_dog.h5')

# Define a function to make a prediction on a new image
def predict_image_class(img):
    # Preprocess the image
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Make a prediction using the model
    prediction = model.predict(x)

    # Print the predicted class label (cat or dog)
    if prediction < 0.5:
        return "This is a cat!"
    else:
        return "This is a dog!"

# Define the Streamlit app
def main(): 
    # Set the title of the app
    st.title("Cat or Dog Classifier")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, make a prediction and display the result
    if uploaded_file is not None:
        # Load the image from the uploaded file
        img = image.load_img(uploaded_file, target_size=(150, 150))

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Make a prediction using the model and display the result
        result = predict_image_class(img)
        st.write(result)

if __name__ == '__main__':
    main()
