"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
from streamlit_option_menu import option_menu
import joblib,os
from pathlib import Path
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer_JM2.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    image = Image.open("resources/imgs/logo.jpg")
    st.image(image)
    st.title("Green'r Foot Classifier \n _____________________________________________________")
    #st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    with st.sidebar:
        selected = option_menu("Menu", ["Home", "About Us", "Predictions", "Contact Us"], 
        icons=['house', 'info-circle', 'graph-up-arrow', 'telephone'], menu_icon="menu-button", default_index=1)
    selected

    # Building out the "Information" page
    if selected == "Home":
        st.subheader("Walk a Greener Path")
        
        # You can read a markdown file from supporting resources folder
        st.markdown("Consumers gravitate toward companies that are built around lessening one’s environmental impact or carbon footprint. Green'r Foot provides an accurate and robust solution that gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories, thus increasing their insights and informing future marketing strategies.")
        st.markdown("Choose Green'r Foot and walk a greener path.")

        #st.subheader("Raw Twitter data and label")
        #if st.checkbox('Show raw data'): # data is hidden if box is unchecked
        #    st.write(raw[['sentiment', 'message']]) # will write the df to the page
        
        
    if selected == "About Us":
        
        st.header("About Green'r Foot Classifier")
        st.markdown("")
        # Creating a text box for user input
        #tweet_text = st.text_area("Enter Text","Type Here")

        #if st.button("Classify"):
            # Transforming user input with vectorizer
         #   vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
          #  predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
          #  prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
        #    st.success("Text Categorized as: {}".format(prediction))



    # Building out the predication page
    if selected == "Predictions":
        st.markdown("This process is fairly easy and user friendly. All you have to do is enter a text (ideally a tweet relating to climate change) and it will be classified according to it's sentiment - whether it shows belief in climate change or not. \n See below on how to interpret results.")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/SVC_linear_JM2.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Sentiment: {}".format(prediction))
            
        with st.expander("ℹ️ How to interpret the results", expanded=False):
            st.write(
            """
            Sentiment is categorized into 4 classes:\n
            [-1] = **Anti**: the tweet does not believe in man-made climate change \n
            [0] = **Neutral**: the tweet neither supports nor refutes the belief of man-made climate change \n
            [1] = **Pro**: the tweet supports the belief of man-made climate change \n
            [2] = **News**: the tweet links to factual news about climate change \n
        
            """
        )
        st.write("")
        
        st.info("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
        
        #st.subheader("Raw Twitter data and label")
        #if st.checkbox('Show raw data'): # data is hidden if box is unchecked
        #    st.write(raw[['sentiment', 'message']]) # will write the df to the page
    
    # Building out the predication page
    if selected == "Contact Us":
        image2 = Image.open('resources/imgs/meet-the-team.png')
        st.image(image2)
        


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
