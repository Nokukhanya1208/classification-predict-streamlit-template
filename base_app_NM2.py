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
from PIL import Image

# Data dependencies
import pandas as pd 
import time

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file
svc_vect = open("resources/vectorizer_JM2_st.pkl", "rb")
tweet_vect = joblib.load(svc_vect)

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages

    st.title("Green'r Foot  \n _____________________________________________________")
    

    # Creating sidebar using streamlit-option-menu 
    
    with st.sidebar:
        selected = option_menu("Menu", ["Home", "Raw Data", "Predictions", "About Us",  "Contact Us"], 
        icons=['house', 'table', 'graph-up-arrow', 'info-circle', 'telephone'], menu_icon="menu-button", default_index=0)


    # Building out the "Home" page
    if selected == "Home":
        image = Image.open("resources/imgs/logo4.jpg")
        st.image(image)
        
        st.subheader("Tweet Classifier")
        st.markdown("Consumers gravitate toward companies that are built around lessening one’s environmental impact or carbon footprint. Green'r Foot provides an accurate and robust solution that gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories, thus increasing their insights and informing future marketing strategies.")
        st.markdown("Choose Green'r Foot and walk a greener path.")

        
    # Building out the raw data page
    if selected == "Raw Data":
        tab1, tab2 = st.tabs(["Data description", "Data Visualizations"])
        with tab1:
            
            st.markdown("The collection of the raw data was funded by a Canada Foundation for Innovation JELF Grant to Chris Bauch, University of Waterloo.")
            st.write(
            """
            This dataset aggregates tweets pertaining to climate change collected between Apr 27, 2015 and Feb 21, 2018. In total, 43943 tweets were annotated. Each tweet is labelled independently by 3 reviewers. This dataset only contains tweets that all 3 reviewers agreed on (the rest were discarded). \n
            Each tweet is labelled as one of the following classes: \n
            * 2(News): the tweet links to factual news about climate change \n
            * 1(Pro): the tweet supports the belief of man-made climate change \n
            * 0(Neutral): the tweet neither supports nor refutes the belief of man-made climate change \n
            * -1(Anti): the tweet does not believe in man-made climate change

            """
            )
            st.write("")
            
        with tab2:
            

            if st.checkbox("Show sentiment value count"):
                image2 = Image.open("resources/imgs/Sentiment.png")
                st.image(image2)
            
            if st.checkbox("Show raw data"):
                job_filter = st.selectbox("Select sentiment", pd.unique(raw['sentiment']))
           
           
                # creating a single-element container.
                placeholder = st.empty()
           
                # dataframe filter 
           
                df = raw[raw['sentiment']==job_filter]
            
                for seconds in range(100):
                #while True: 
                                   
                   with placeholder.container():       
            
                  
                       st.markdown("### Raw data")
                       st.dataframe(df)
                       time.sleep(1)
                   
            if st.checkbox("Show raw data word cloud"):
                image5 = Image.open("resources/imgs/100commonwordsrawdata.png")
                st.image(image5)


    # Building out the predications page
    if selected == "Predictions":
        st.subheader("How It Works")
        st.markdown("Click on a tab to choose your desired classifier then enter a tweet relating to climate change and it will be classified according to its sentiment. \n See below on how to interpret results.")
        #using tabs for different predictors
        tab1, tab2 = st.tabs(["Support Vector Classifier", "Logistic Regression Classifier"])
        with tab1:
            
            st.markdown("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
        
            with st.expander("🐤 Tweets", expanded=False):
                st.write(
                """
                * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                Expected output = -1 \n
                * Polar bears for global warming. Fish for water pollution.\n
                Expected output = 0 \n
                * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                Expected output = 1 \n
                * G20 to focus on climate change despite Trump’s resistance \n
                Expected output = 2
        
                """
            )
            st.write("")
            
            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text Below")

            if st.button("SVC Classify"):
            # Transforming user input with vectorizer
                vect_text = tweet_vect.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join("resources/SVC_linear_Final_NM.pkl"),"rb"))
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
        
        
        
        with tab2:
            
            st.markdown("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
        
            with st.expander("🐤 Tweets", expanded=False):
                st.write(
                """
                * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                Expected output = -1 \n
                * Polar bears for global warming. Fish for water pollution.\n
                Expected output = 0 \n
                * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                Expected output = 1 \n
                * G20 to focus on climate change despite Trump’s resistance \n
                Expected output = 2
        
                """
            )
            st.write("")
            
            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text")

            if st.button("LRC Classify"):
            # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
                predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
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
        
        
            
    # Building out the "About Us" page
    if selected == "About Us":
        # Using Tabs
        tab1, tab2, tab3 = st.tabs(["About Green'r Foot", "Our Classifier", "Meet the Team"]) 
        with tab1:
            
            st.markdown("We’re proud to be an industry leader in promoting eco-friendly business practices. Striving to protect and sustain our environment is a given at every stage of our service lifecycles.\n  Our green vision goes beyond delivering artificial intelligence services. Helping customers sustain their businesses is at the core of our mission as a green company. We design innovative technology to help businesses save time, reduce costs, and make better business decisions to ensure their footprint is greener.")
            st.markdown("Green'r Foot's mission is to accelerate the world’s transition to sustainable energy, ensuring that even through humanity's advances, Earth is still preserved for generations to come. Our data science consultants deliver incredible value by evaluating and recommending strategic business decisions to further your organizational ambitions.")
            image4 = Image.open("resources/imgs/NWLD-composite-EarthDay-km.jpg")
            st.image(image4)
        with tab2:
            st.write(
            """
            Our Tweet Classifier app gives you a variety of Machine Learning Models to choose from. The models selected showed high performance over the others with a sentiment claassification accuracy of over 80%. \n 
            Our leading model is the Support Vector Classifier with an impressive accuracy of over of 90%, ensuring our users accuracy that will inform great business decisions.
    
            """
            )
            image3 = Image.open("resources/imgs/Sentiment-notebook-picture.jpg")
            st.image(image3)
        with tab3:
            image5 = Image.open("resources/imgs/team-picture.png")
            st.image(image5)
     
    
    # Building out the contact page
    if selected == "Contact Us":
        with st.form("form1", clear_on_submit=True):
            st.subheader("Get in touch with us")
            name = st.text_input("Enter full name")
            email = st.text_input("Enter email")
            message = st.text_area("Message")
            
            submit = st.form_submit_button("Submit Form")
            if submit:
                st.write("Your form has been submitted and we will be in touch 🙂")
        


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
