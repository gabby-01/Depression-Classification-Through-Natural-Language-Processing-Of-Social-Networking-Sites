# Library Installation
# -- streamlit==1.22.0
# -- tensorflow==2.10.1
# -- matplotlib
# -- seaborn
# -- nltk
# -- wordninja
# -- spacy

# Library Import
import streamlit as st # Deployment software for machine learning
import pandas as pd # Data manipulation and analysis
import altair as alt #
import tensorflow as tf # Build and train deep learning models
import matplotlib.pyplot as plt # Create static, animated, and interactive visualizations
import matplotlib #
import re # Regular expression library
import nltk # Toolkit build for working with NLP
import wordninja  # Probabilistically split concatenated words using NLP based on English Wikipedia uni-gram frequencies.
import spacy # For advanced NLP
import pickle # Serialize and de-serialize a Python object structure
import sqlite3 # Database storage
import datetime # Current date & time
matplotlib.use('Agg') #
import seaborn as sns #
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tensorflow.keras.models import load_model # type: ignore # Load the saved model
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore # Keras tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore # Pad all the sequences to have the same length by adding zeros to the sequences
from nltk.corpus import stopwords # Access the stopwords from nltk corpus
from nltk.tokenize import word_tokenize # Module for tokenize words
from datetime import datetime # Retrieve the current date & time

# Download necessary NLTK packages
nltk.download('punkt') # Models for tokenizing sentences and texts into individual words
nltk.download('stopwords') # Package contains lists of stop words
spacy.cli.download('en_core_web_sm') # Download spacy model

# Connect database
conn = sqlite3.connect('data.db')
c = conn.cursor()

@st.cache(allow_output_mutation=True)
def load_spacy_model():
    return spacy.load('en_core_web_sm')
nlp = load_spacy_model()

# Load the pre-trained Tokenizer
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as file:
        return pickle.load(file)
tokenizer = load_tokenizer()

# Load the HDF5 BiLSTM model
@st.cache(allow_output_mutation=True)
def load_bilstm_model():
    return load_model('./BiLSTM_model.h5')
BiLSTM_model = load_bilstm_model()

# Load the Hugging Face pre-trained model
@st.cache(allow_output_mutation=True)
def load_huggingface_model():
    tokenizer = AutoTokenizer.from_pretrained("ShreyaR/finetuned-roberta-depression")
    model = AutoModelForSequenceClassification.from_pretrained("ShreyaR/finetuned-roberta-depression")
    return tokenizer, model
hf_tokenizer, hf_model = load_huggingface_model()

# Load the images
Depression_Home_1 = "Pictures\Depression_Home_1.png"
Depression_Home_2 = "Pictures\Depression_Home_2.png"
Depression_Home_3 = "Pictures\Depression_Home_3.png"
Depression_Home_4 = "Pictures\Depression_Home_4.png"
Depression_Home_5 = "Pictures\Depression_Home_5.png"
Depression_Home_6 = "Pictures\Depression_Home_6.png"
Depression_Home_7 = "Pictures\Depression_Home_7.png"
Depression_Depression_Detection_8 = "Pictures\Depression_Depression_Detection_8.png"

# Text preproessing techniques
# Function to remove emojis 
def remove_emojis_and_count(text):
    emoji_pattern = re.compile(
        '['
        '\U0001F1E0-\U0001F1FF'  # Flags (iOS)
        '\U0001F300-\U0001F5FF'  # Symbols & Pictographs
        '\U0001F600-\U0001F64F'  # Emoticons
        '\U0001F680-\U0001F6FF'  # Transport & Map Symbols
        '\U0001F700-\U0001F77F'  # Alchemical Symbols
        '\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        '\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
        '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        '\U0001FA00-\U0001FA6F'  # Chess Symbols
        '\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        '\U00002702-\U000027B0'  # Dingbats
        '\U000024C2-\U0001F251'  # Miscellaneous symbols and pictographs
        ']+', flags=re.UNICODE)
    # Remove emojis from the text
    cleaned_text = emoji_pattern.sub(r'', text)
    # Return cleaned text 
    return cleaned_text

# Function to remove URLs
def remove_urls_and_count(text):
    # Regular expression to identify URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    # Remove URLs from the text
    cleaned_text = url_pattern.sub(r'', text)
    # Return cleaned text
    return cleaned_text

# Function to remove emails
def remove_emails_and_count(text):
    # Regular expression to identify emails
    email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    # Remove emails from the text
    cleaned_text = email_pattern.sub(r'', text)
    # Return cleaned text
    return cleaned_text

# Function to convert text into lowercase
def convert_to_lowercase(text):
    # Convert text to lowercase
    lowercased_text = text.lower()
    # Return the lowercase text
    return lowercased_text

# Function to replace the curly apostrophe to straight apostrophe
def normalize_symbols(text):
    # Replace specific apostrophe symbols with standard ones
    normalized_text = text.replace("’", "'")
    # Return the normalized text
    return normalized_text

# Function to replace abbreviations
abbreviation_list = {
  "ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have", "didn't": "did not",
  "doesn't": "does not","don't": "do not","dont": "do not","gonna": "going to","hadn't": "had not","hadn't've": "had not have", "hasn't": "has not","haven't": "have not","he'd": "he would","idk": "i do not know",
  "he'd've": "he would have","he'll": "he will",  "he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","i'd": "i would","i'd've": "i would have",
  "i'll": "i will","i'll've": "i will have","i'm": "i am","im": "i am","i've": "i have","isn't": "is not","it'd": "it had","it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is", 
  "irl": "in real life","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not",
  "mustn't've": "must not have","needn't": "need not","needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
  "shan't've": "shall not have","she'd": "she would","she'd've": "she would have", "she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have","shouldn't": "should not",
  "shouldn't've": "should not have","so've": "so have","so's": "so is","that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have",
  "there's": "there is","they'd": "they would","they'd've": "they would have","they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have","wasn't": "was not",
  "we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what'll've": "what will have",
  "what're": "what are","what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have", "who'll": "who will",
  "who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is","why've": "why have","will've": "will have","won't": "will not","won't've": "will not have","would've": "would have",
  "wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
  "you'd": "you had","you'd've": "you would have", "you'll": "you you will","you'll've": "you you will have","you're": "you are","you've": "you have",
}
def replace_abbreviations(text):
    for abb, replacement in abbreviation_list.items():
        text = text.replace(abb, replacement)
    return text

# Function to tokenize text
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# Function to segment words in text
def segment_words(text):
    # Combine the text into a single string if it's a list of tokens
    if isinstance(text, list):
        text = " ".join(text)
    # Use wordninja to split concatenated words into a list of words
    segmented_words = wordninja.split(text)
    # Return the list of segmented words
    return segmented_words

# Function to remove stop words
stop_words = set(stopwords.words('english')) # Load the list of stop words
def remove_stop_words(tokens):
    # Remove stop words
    tokens_without_stopwords = [token for token in tokens if token.lower() not in stop_words]
    return tokens_without_stopwords

# Function to remove punctuations
def remove_punctuation(tokens):
    tokens_without_punctuation = [token for token in tokens if token.isalnum()]
    return tokens_without_punctuation

# Function to remove numbers
def remove_numbers(tokens):
    tokens_without_numbers = [token for token in tokens if not token.isdigit()]
    return tokens_without_numbers

# Function to lemmatize words
def lemmatize_tokens(tokens):
    # Join the tokens back into a sentence
    text = ' '.join(tokens)
    # Process the text using spacy
    doc = nlp(text)
    # Lemmatize each token and return the lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

# Define preprocessing functions
def preprocess_text(text):
    # Normalize symbols and remove emojis, urls, emails
    text = normalize_symbols(text)
    text = remove_emojis_and_count(text)
    text = remove_urls_and_count(text)
    text = remove_emails_and_count(text)

    # Convert text to lowercase
    text = convert_to_lowercase(text)

    # Replace abbreviations
    text = replace_abbreviations(text)

    # Tokenize text and remove stop words and numbers
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = remove_numbers(tokens)

    # Segment words and remove punctuation
    segmented_words = segment_words(tokens)
    segmented_words = remove_punctuation(segmented_words)

    # Lemmatize tokens
    lemmatized_tokens = lemmatize_tokens(segmented_words)

    # Reconstruct the processed text
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Function to convert text into sequences
def text_to_sequences(text):
    # Apply the text preprocessing techniques
    processed_text = preprocess_text(text)

    # Use the tokenizer to transform the texts to sequences
    sequence = tokenizer.texts_to_sequences([processed_text])

    # Pad the sequences to ensure the texts are all same length
    padded_sequence = pad_sequences(sequence, maxlen = 467, padding = 'post')
    return padded_sequence

# Function to predict with Hugging Face model
def predict_with_huggingface_model(text):
    inputs = hf_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=467)
    outputs = hf_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities[0][1].item(), probabilities[0][0].item()

# Function to create database table
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT, prediction TEXT, depression_proba TEXT, non_depression_proba TEXT, postdate DATE)')
    conn.commit()

# Function to add data into database table
def add_data(message, prediction, depression_proba, non_depression_proba, postdate):
    c.execute('INSERT INTO predictionTable(message, prediction, depression_proba, non_depression_proba, postdate) VALUES(?,?,?,?,?)', (message, prediction, depression_proba, non_depression_proba, postdate))
    conn.commit()

# Function to view all the data from the database table
@st.cache
def view_all_data():
    c.execute("SELECT * FROM predictionTable")
    return c.fetchall()

# Construct the website application
def main():
    # Define the 4 pages: Home, Depression Detection, Dashboard, About
    menu = ["Home", "Depression Detection", "Dashboard", "About"]
    create_table() 
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.markdown("# Home🏠")
        st.image(Depression_Home_1, caption = "Unveiling the Tapestry of Emotions: Understanding Depression", use_column_width = True)
        st.markdown("""
                    **Depression is more than just feeling sad** or going through a rough patch; it's a **serious mental condition that affects millions world wide**. Like a shadow,
                    it can silently weave its way through the tapestry of our lives, often hidden beneath the surface. It is charactirized by a **profound sense of despair, a deep-rooted
                    sense of hopelessness, and a loss of interest** in the world that once brought joy.

                    In this realm, the simplest of tasks become mountains to climb, and the world often seems devoid of color. Beyond a singular feeling, depression encompasses a range
                    of **physical and emotional problems**, manifesting differently from person to person. While the **exact cause is unknown**, a blend of genetic, biological, environmental,
                    and psychological factors** contributes to its onset.
                    
                    #### Symptoms of Depression:
                    - **A persistent sad, anxious, or "empty" mood**
                    - **Feelings of hopelessness or pessimism**
                    - **Irritability**
                    - **Loss of interest** in hobbies and activities
                    - **Decreased energy or fatigue**
                    - **Moving or talking more slowly**
                    - **Feeling restless** or having trouble sitting still
                    - **Difficulty concentrating**, remebering, or making decisions
                    - **Difficulty sleeping**, early-morning awakening, or oversleeping
                    - **Appetite and/or weight changes**
                    - **Thoughts of death or suicide**, or suicide attempts
                    - **Aches or pains**, headaches, cramps, or digestive problems without a clear physical cause and/or that do not ease even with treatment
                    
                    #### Prevalance of Depression:
                    """)
        # Define custom styles for tabs to look like boxes
        tab_style = """
        <style>
            /* Tabs container */
            .stTabs > div {
                box-shadow: none !important;  /* Remove default underline */
            }
            /* Individual tab */
            .tabs {
                border: 2px solid #e1e4e8 !important; /* Tab border */
                border-radius: 12px 12px 0 0 !important; /* Rounded corners on the top */
                margin-right: 8px; /* Spacing between tabs */
            }
            /* Individual tab background and padding */
            button[data-baseweb="tab"] {
                background-color: #fff !important; /* Tab background color */
                padding: 12px 24px !important; /* Tab padding */
                box-shadow: none !important; /* Remove button shadow */
                border: none !important; /* Remove button border */
                font-size: 16px !important; /* Increase font size */
                font-weight: 500 !important; /* Adjust font weight */
                color: #000 !important; /* Tab text color */
                border-radius: 12px 12px 0 0 !important; /* Rounded corners on the top */
            }
            /* Active tab */
            button[data-baseweb="tab"][aria-selected="true"] {
                background-color: #000 !important; /* Active tab background color to black */
                color: #fff !important; /* Active tab text color to white */
                border-bottom: 2px solid white !important; /* Hide the bottom border for active tab */
            }
            /* Tab hover effect */
            button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
                background-color: #eee !important; /* Tab hover background color */
            }
            /* Tab focus effect */
            button[data-baseweb="tab"]:focus {
                outline: none !important; /* Remove focus outline */
                box-shadow: 0 0 0 0.2rem rgba(0, 0, 0, 0.5) !important; /* Custom focus shadow to match brand color */
            }
        </style>
        """
        # Inject custom styles into the app
        st.markdown(tab_style, unsafe_allow_html = True)

        # Define the tabs: Region, Gender, Mortality, Age
        tab1, tab2, tab3 = st.tabs(["Region", "Age & Gender", "Mortality"])
        with tab1:
            col1, col2 = st.columns([2.7, 1])
            with col1:
                st.image(Depression_Home_2, caption = "Prevalence of Depression by Region", use_column_width = True)
            with col2:
                st.image(Depression_Home_3, caption = "Top 6 Regions", use_column_width = True)
                st.image(Depression_Home_4, caption = "Top 9 Regions' Universities", use_column_width = True)
            st.markdown("""
                        ##### The Global Tapestry of Depression: A Regional Overview
                        Depression is a global challenge that transcends borders, cultures, and economies, affecting individuals from all walks of life. The prevalence of this condition
                        varies across regions, painting a diverse picture of its impact on societies worldwide.

                        **The World Health Organization estimates that over 322 million people live with depression, a condition that constitutes a significant portion of the global 
                        burden of disease.** The map and charts presented here offer a snapshot of diagnosed clinical depression rates by region, providing insight into the geographical
                        distribution of this pervasive mental health issue.
                        - The **African Region** shows a considerable variation in depression rates, reflecting the complex interplay between socioeconomic factors and access to mental health 
                        services.
                        - In the **Eastern Mediterranean Region**, cultural factors and conflict may influence the reported prevalence, with some areas seeing higher rates than others.
                        - The **European Region** experiences a relatively high prevalence, possibly due to better detection and reporting systems alongside lifestyle and environmental factors.
                        - The **Region of the Americas** presents a broad range of prevalence rates, suggesting a strong influence of cultural diversity and economic disparity.
                        - In the **South-East Asia Region**, there is a growing recognition of mental health issues, though stigma and healthcare access remain significant barriers.
                        - The **Western Pacific Region** illustrates varied rates, highlighting the challenges in addressing mental health within diverse cultural contexts.
                        
                        Universities across the globe are microcosms reflecting broader societal challenges, and the prevalence of depression among students is a growing concern.
                        **Countries like Malaysia, Ethiopia, and Thailand show significant differences in these rates**, indicative of the diverse academic pressures, social environments, 
                        and mental health resources available to students. In Malaysia, 27.5% of university students reported experiencing depression, a figure that underscores the importance 
                        of supportive services and mental health awareness on campus. Ethiopia and Thailand, with prevalences of 21.6% and 47.01% respectively, highlight the varying degrees of 
                        mental health challenges faced by students in different educational and cultural contexts. These figures illustrate the urgent need for universities to prioritize mental 
                        health, providing comprehensive support systems to foster the well-being of their students.
                        """)
        with tab2:
            st.image(Depression_Home_5, caption = "Prevalence of Depression by Age & Gender", use_column_width = True)
            st.markdown("""
                        ##### The Intersection of Age and Gender in Depression Prevalence
                        The prevalence of depression varies not just by region and university attendance, but also significantly across different age groups and between genders. Our analysis 
                        reveals poignant differences that shed light on how depression impacts individuals throughout their lifespan.
                        
                        **For women**, the data suggests a concerning trend: **nearly one in three women experience depression by the time they reach 65**. The figures climb from 24.5% in the 18–25 
                        age group to a peak of 33.2% in those aged 50–64. This pattern underscores the vital need for targeted mental health support for women, particularly as they approach middle age.
                        
                        Conversely, men experience depression differently. By the age of 65, around one in five men report experiencing depression. Rates increase from 15% in young adulthood (18-25) to 
                        19.4% in the 50–64 age bracket. This information is crucial, highlighting that while men report depression at a lower rate than women, it remains a significant issue that needs 
                        addressing.
                        
                        These statistics illuminate the stark realities of depression's prevalence and emphasize the importance of gender-specific research and support mechanisms. They also serve as a 
                        reminder that depression is a pervasive mental health challenge across the lifespan, necessitating continued awareness, early intervention, and accessible treatment options for 
                        all individuals, regardless of age or gender.
                        """)
        with tab3:
            col1, col2 = st.columns([1.17, 1])
            with col1:
                st.image(Depression_Home_6, caption = "Mortality Rate by Age", use_column_width = True)
            with col2:
                st.image(Depression_Home_7, caption = "Mortality Rate by Gender", use_column_width = True)
            st.markdown("""
                        ##### Mortality Rates by Age and Gender: A Closer Look
                        The relationship between depression and mortality is complex, with significant variances across different age groups and between genders. Our latest data illustrates how mortality 
                        rates due to depression-related issues, including suicide, shift and change throughout the lifespan.
                        
                        The first chart reveals a **consistent pattern across all age groups**, with the highest age-adjusted mortality rates seen in those aged 85 or older. This suggests that the oldest 
                        populations are the most vulnerable, emphasizing the need for heightened mental health support and interventions for our elderly.
                        
                        Among the younger population, particularly those **less than 14 years of age**, the rates are thankfully lower, but even a single life lost to depression is one too many, highlighting 
                        the importance of early mental health education and support.
                        
                        In dissecting the data further by gender, the second chart presents a startling disparity. **Males have a markedly higher mortality rate** in most age groups compared to females, with a 
                        notable peak in middle age. This indicates a potential crisis point where targeted preventative measures could be the most beneficial.
                        
                        Conversely, females show a relatively steady rate, with a slight increase in the older age groups. While lower than males, the impact on females is nonetheless significant and warrants 
                        dedicated resources and attention.
                        """)
    if choice == "Depression Detection":
        st.markdown("# Depression Detection 🔍")
        st.image(Depression_Depression_Detection_8, use_column_width = True)
        # Using CSS to further customize the appearance of the UI
        st.markdown(
            """
            <style>
            .big-font {
                font-size:20px !important;  /* Increase font size for better readability */
                font-weight: bold;  /* Bold font weight */
                color: #333;  /* Dark grey for a more formal appearance */
            }
            .title-font {
                font-size:24px !important;  /* Even larger font size for title */
                font-weight: bold;
                margin-bottom: 0px !important;
            }
            .stButton > button {
                width: 100%;
                padding: 16px 24px;  /* Slightly more padding for aesthetic */
                border: 2px solid black;  /* Black border */
                background-color: black;  /* Black background */
                color: white;  /* White text */
                font-size: 18px;  /* Larger font size */
                font-weight: bold;  /* Bold text */
                border-radius: 4px;  /* Rounded corners */
                transition: transform 0.1s, box-shadow 0.1s;  /* Smooth transition for press and hover effects */
            }
            .stButton > button:hover {
                transform: scale(1.02);  /* Slightly enlarge button on hover */
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);  /* Add shadow for depth */
            }
            .stButton > button:active {
                transform: scale(0.98);  /* Shrink button when pressed */
                box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);  /* Smaller shadow for pressed effect */
            }
            .stTextArea {  /* Target the text area container */
                margin-top: -20px !important;  /* Move the text area up to reduce gap */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        with st.form(key = 'mlform'):
            col_detection1, col_detection2 = st.columns([3, 1])
            with col_detection1:
                st.markdown('<p class="title-font">Depression Detection System</p>', unsafe_allow_html = True)
                message = st.text_area("", placeholder="Type your text here...", height = 150)
                submit_message = st.form_submit_button(label = 'Predict')
            with col_detection2:
                st.markdown('##')  # This markdown is added just to align the column content at the top
                st.markdown('<p class="big-font">This Depression Detection system will predict text with a max length of 467 words as <strong>Depression</strong> or <strong>Non-Depression</strong>.</p>', unsafe_allow_html=True)
        
        if submit_message:
            probability_of_depression, probability_of_non_depression = predict_with_huggingface_model(message)
            result = "Depression" if probability_of_depression > 0.5 else "Non-Depression"
            postdate = datetime.now()
            add_data(message, result, probability_of_depression, probability_of_non_depression, postdate)
            st.success("Data Submitted")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("**Original Text**")
                st.markdown("""
                <style>
                    .justify-text p {
                        text-align: justify;
                        text-justify: inter-word;
                    }
                </style>
                <div class="justify-text"><p>""" + message + """</p></div>
            """, unsafe_allow_html=True)

            with res_col2:
                st.info("**Prediction**")
                st.write(result)

                st.info("**Probability**")
                prob_col1, prob_col2 = st.columns(2)  # Create two columns for a side-by-side layout
                with prob_col1:
                    st.metric("**Depression**", f"{probability_of_depression:.2%}")
                with prob_col2:
                    st.metric("**Non-Depression**", f"{probability_of_non_depression:.2%}")

                # Prepare data for plotting
                data = pd.DataFrame({
                    'Class': ['Depression', 'Non-Depression'],
                    'Probability': [probability_of_depression, probability_of_non_depression]
                })
                
                # Custom color palette
                custom_palette = sns.color_palette("magma")

                # Plotting the bar chart with custom colors using seaborn
                sns.set_theme(style="whitegrid")
                fig, ax = plt.subplots()
                barplot = sns.barplot(
                    x='Class', 
                    y='Probability', 
                    data=data, 
                    palette=custom_palette,
                    ax=ax
                )

                # Customizing the bar labels
                for p in barplot.patches:
                    barplot.annotate(format(p.get_height(), '.2%'), 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha = 'center', va = 'center', 
                                    xytext = (0, 9), 
                                    textcoords = 'offset points')

                # Hide the left, top, and right spines
                sns.despine()

                # Show the plot
                st.pyplot(fig)

    if choice == "Dashboard":
        st.markdown("# Dashboard 📊")
        st.markdown(
            """
            <div style="
                background-color: black; 
                color: white; 
                padding: 10px; 
                text-align: center; 
                font-size: 24px; 
                border-radius: 10px;
                box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
                font-family: 'Gill Sans';
                margin: 30px 0;">
                Prediction History
            </div>
            """, 
            unsafe_allow_html=True
        )
        # Dashboard 1
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data, columns = ['Message', 'Prediction', 'Depression Probability', 'Non-Depression Probability', 'Date_Time'])
        # Format the "Probability" columns to show as percentages with 2 decimal places
        new_df['Depression Probability'] = pd.to_numeric(new_df['Depression Probability']).map("{:.2%}".format)
        new_df['Non-Depression Probability'] = pd.to_numeric(new_df['Non-Depression Probability']).map("{:.2%}".format)
        new_df['Date_Time'] = pd.to_datetime(new_df['Date_Time']).dt.strftime('%Y-%m-%d %H:%M')
        # Style the DataFrame
        def highlight_predictions(val):
            color = 'red' if val == 'Depression' else 'green'
            return f'background-color: {color}; color: white'
        st.dataframe(new_df.style.applymap(highlight_predictions, subset=['Prediction']), width=700, height=350)

        # Dashboard 2
        st.markdown(
            """
            <div style="
                background-color: black; 
                color: white; 
                padding: 10px; 
                text-align: center; 
                font-size: 24px; 
                border-radius: 10px;
                box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
                font-family: 'Gill Sans';
                margin: 30px 0;">
                Prediction Dashboard
            </div>
            """, 
            unsafe_allow_html=True
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            # Count the frequency of each prediction
            prediction_counts = new_df['Prediction'].value_counts().rename('Frequency')
            # Create a color map based on the prediction categories
            colors = ['green' if x == 'Non-Depression' else 'red' for x in prediction_counts.index]
            # Plot with Matplotlib
            fig, ax = plt.subplots()
            prediction_counts.plot(kind='bar', ax=ax, color=colors)  # Apply the colors to the bars
            ax.set_xlabel('Prediction')
            ax.set_ylabel('Frequency')
            ax.set_title('Frequency of Predictions')
            ax.tick_params(axis='x', rotation=0)
            st.pyplot(fig)

        with c2:
            # Calculate the maximum length of words in the messages
            max_length = new_df['Message'].apply(lambda x: len(x.split())).max()
            # Display the maximum length in a styled box with a title
            st.markdown(
                """
                <div style="
                    text-align: center;
                    margin: 8px;">
                    <h6 style="margin:0; padding:0;">Maximum Length Text</h6>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    border: 2px solid black;
                    padding: 15px;
                    text-align: center;
                    font-size: 20px;
                    border-radius: 10px;
                    box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
                    font-family: 'Gill Sans';
                    margin-top: 25px;">
                    {max_length} words
                </div>
                """, 
                unsafe_allow_html=True
            )
            # Calculate the minimum length of words in the messages
            min_length = new_df['Message'].apply(lambda x: len(x.split())).min()
            # Display the maximum length in a styled box with a title
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown(
                """
                <div style="
                    text-align: center;
                    margin: 8px;">
                    <h6 style="margin:0; padding:0;">Minimum Length Text</h6>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div style="
                    border: 2px solid black;
                    padding: 15px;
                    text-align: center;
                    font-size: 20px;
                    border-radius: 10px;
                    box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);
                    font-family: 'Gill Sans';
                    margin-top: 25px;">
                    {min_length} words
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Plot 3
        stored_data = view_all_data()
        # Custom color palette
        custom_palette_1 = sns.color_palette("Blues")
        new_df = pd.DataFrame(stored_data, columns=['Message', 'Prediction', 'Depression Probability', 'Non-Depression Probability', 'Date_Time'])
        # Ensure 'Date_Time' is a datetime type
        new_df['Date_Time'] = pd.to_datetime(new_df['Date_Time'])
        # Extract the date part only
        new_df['Date'] = new_df['Date_Time'].dt.date
        # Count the number of messages per date
        date_counts = new_df.groupby('Date').size().rename('Counts')
        # Ensure the index is a DatetimeIndex for formatting
        date_counts.index = pd.to_datetime(date_counts.index)
        # Plot with matplotlib
        fig, ax = plt.subplots()
        date_counts.plot(kind='bar', ax=ax, width=0.2, color = custom_palette_1)  # 'width' controls the bar width
        ax.set_xlabel('Date')
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency of Input Text By Date')
        ax.set_xticklabels(date_counts.index.strftime('%Y-%m-%d'), rotation=90, ha='right')
        ax.legend().set_visible(False)
        # Use Streamlit to show the plot
        st.pyplot(fig)

    if choice == "About":
        st.markdown("# About")
        st.markdown("""
                    #### Welcome to Our Text-based Depression Detection System!
                    This tool leverages advanced **Natural Language Processing (NLP)** techniques to analyze posts from social networking sites, aiming to detect signs of depression. Our system is 
                    designed to aid mental health professionals by providing an additional layer of data-driven insight, helping to identify individuals who may benefit from further psychological assessment or intervention.
                    
                    #### How It Works
                    Our system processes textual data using a sophisticated machine learning model trained on a diverse dataset of social media posts. These posts have been annotated for depressive markers based on psychological 
                    research, ensuring that our model learns to recognize subtle cues of depressive sentiment. Once deployed, the model evaluates new inputs, categorizing them based on the likelihood of depressive content.
                    
                    #### Why It Matters
                    Depression is a common but serious mood disorder that affects many people around the world. Early detection is crucial for effective treatment, but many individuals do not seek help due to stigma or lack of 
                    awareness. By providing a tool that can detect potential signs of depression early and discreetly, we aim to contribute to broader efforts in mental health care, potentially encouraging users to seek professional help sooner.
                    
                    #### Our Commitment
                    We are committed to improving mental health care through technology. This system is continuously refined and updated to improve its accuracy and usability. We adhere to strict data privacy and ethical standards to ensure that 
                    user data is protected and the tool is used responsibly.
                    ##
                    ##
                                                     
                    **Date**: 20/04/2024    
                    **Inventor**: Liew Jun Yen 
                    """)

if __name__ == '__main__':
    main()



