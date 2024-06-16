![Project Poster-1](https://github.com/LiewJunYen-DataAnalyst/Depression-Classification-Through-Natural-Language-Processing-Of-Social-Networking/assets/130137513/7ad4aaef-9dd6-484a-813e-6debe96c380a)

# Introduction 📖
Depression is a widespread mental disorder that affects people globally, with significant disparities across different genders and regions. Recognizing the value of digital platforms in early detection, this project aims to predict signs of user-generated depressive posts on social networking sites (SNS), enhancing the detection of mental health issues in digital communication spaces. Leveraging advanced Natural Language Processing (NLP) models and extensive SNS data, this research addresses this critical health concern effectively.

# Aim & Objectives 🎯
**Aim**: To predict signs of user-generated depressive posts on SNS, enhancing the detection of mental health issues in digital communication spaces.

**Objectives**:
* To generate a dataset by collecting 30,000 user-generated depressive and non-depressive posts from various social networking sites.
* To perform a hybrid approach for data labelling that integrates manual labelling and transfer learning.
* To implement and evaluate four deep learning models.
* To deploy the most effective model with at least 80% predictive accuracy into a web application.

# Dataset 🛢️
The dataset is manually collected by the owner from Facebook, Twitter, and Reddit. It consists of a total of 41,273 samples and can be retrieved via the provided [Data Source.zip](Data%20Source.zip).

# Data Collection Tool 🛠️
* **Twitter API v2:** Used for collecting posts from Twitter.
* **Postman Collection:** Utilized for managing API requests and responses.

# Deep Learning Models Implementation 🤖
* Recurrent Neural Network (**RNN**)
* Convolutional Neural Network (**CNN**)
* Long Short-Term Memory (**LSTM**)
* Bidirectional Long Short-Term Memory (**BiLSTM**)
* Gated Recurrent Unit (**GRU**)
* Robustly Optimized Bidirectional Encoder Representations from Transformers Pretraining Approach (**RoBERTa**)

# Streamlit App 💻
* You can access the web application for depression detection here: [Depression Detection App](https://depression-detection-system.onrender.com).

# Project Documentation 📚
For detailed documentation of the project, including methodology, implementation details, results, and more, please refer to the following complete project documentation in a single PDF file here: [Complete Project Documentation](https://1drv.ms/w/s!Ar5gsj8m4olr0AI5b4L13Y_lQG3e?e=VPldao).

# Python Tools & Versions ⚙️
To ensure reproducibility and compatibility, the following versions of tools and libraries were used in this project:
* **Python:** 3.10.11
* **tensorflow:** 2.10.1
* **pandas:** 2.2.2
* **langdetect:** 1.0.7
* **matplotlib:** 3.9.0
* **seaborn** 0.13.2
* **wordcloud:** 1.9.3
* **numpy:** 1.26.4
* **transformers:** 4.41.2
* **scikit-learn:** 1.5.0
* **torch:** 1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
* **nltk:** 3.8.1
* **wordninja:** 2.0.0
* **spacy:** 3.7.5
* **imbalanced-learn:** 0.12.3
* **keras-tuner:** 1.4.7
* **streamlit:** 1.22.0
