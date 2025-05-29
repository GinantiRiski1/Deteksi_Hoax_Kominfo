import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
from wordcloud import WordCloud
import joblib
import os


# Function to clean the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to check login
def check_login(username, password):
    return username == "admin" and password == "123"

# Importing Dataset
df_fake1 = "Dataset/Fake_fix.xlsx"
df_true1 = "Dataset/True_2.csv"  # Changed to CSV

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Save and Load model functions
def save_model(model, model_name):
    model_path = f"{model_name}.pkl"
    joblib.dump(model, model_path)
    st.write(f"Model saved as {model_path}")

def load_model(model_name):
    model_path = f"{model_name}.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.write(f"Model {model_name} loaded.")
        return model
    else:
        st.write(f"Model {model_name} not found.")
        return None

# Main content, hanya ditampilkan setelah login berhasil
if st.session_state.logged_in:
    # Streamlit application title
    st.title("Deteksi Berita Hoax di Indonesia")

    # Importing dataset with st.cache_data
    @st.cache_data(persist=True)
    def load_data():
        df_fake = pd.read_excel(df_fake1, engine='openpyxl')
        df_true = pd.read_csv(df_true1)  # Read CSV directly
        return df_fake, df_true

    df_fake, df_true = load_data()

    # Inserting a column "class" as target feature
    df_fake["class"] = 0
    df_true["class"] = 1

    # Removing last 10 rows for manual testing
    df_fake_manual_testing = df_fake.tail(10).copy()
    df_fake.drop(df_fake.tail(10).index, inplace=True)

    df_true_manual_testing = df_true.tail(10).copy()
    df_true.drop(df_true.tail(10).index, inplace=True)

    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1

    df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
    df_manual_testing.to_csv("manual_testing.csv", index=False)

    # Merging True and Fake Dataframes
    df_merge = pd.concat([df_fake, df_true], axis=0)

    # Removing columns which are not required
    df = df_merge.drop(["title", "subjek", "date"], axis=1)

    # Cleaning the text
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].apply(wordopt)

    # Defining dependent and independent variables
    x = df["text"]
    y = df["class"]

    # Splitting Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # Convert text to vectors
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0)
    }

    # Sidebar menus with CSS customization
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        .sidebar .sidebar-content .sidebar-close-btn {
            color: #000;
        }
        .sidebar .sidebar-content .stSelectbox .stOption:hover {
            background-color: #4caf4f !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    menu = st.sidebar.selectbox("Menu", ["Pemilihan Model", "Deteksi Berita", "Visualisasi", "EDA", "Perbandingan Model"])

    if menu == "Pemilihan Model":
        st.header("Pemilihan Model")

        # Select Model
        model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))

        # Load selected model if exists, otherwise train
        loaded_model = load_model(model_choice)
        if loaded_model:
            selected_model = loaded_model
        else:
            selected_model = models[model_choice]
            selected_model.fit(xv_train, y_train)

        # Train selected model
        pred = selected_model.predict(xv_test)

        # Input Text for detection
        news = st.text_area("Masukkan teks berita:")

        # Detect button
        if st.button("Deteksi"):
            def manual_testing(news):
                testing_news = {"text": [news]}
                new_def_test = pd.DataFrame(testing_news)
                new_def_test["text"] = new_def_test["text"].apply(wordopt)
                new_x_test = new_def_test["text"]
                new_xv_test = vectorization.transform(new_x_test)
                pred = selected_model.predict(new_xv_test)
                return "Berita Hoaks" if pred[0] == 0 else "Berita Real"

            result = manual_testing(news)
            st.write(f"Hasil Deteksi: {result}")

        # Save the model
        if st.button("Save Model"):
            save_model(selected_model, model_choice)

    elif menu == "Deteksi Berita":
        st.header("Deteksi Berita")

        # Select Model
        model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))

        # Load selected model if exists, otherwise train
        loaded_model = load_model(model_choice)
        if loaded_model:
            selected_model = loaded_model
        else:
            selected_model = models[model_choice]
            selected_model.fit(xv_train, y_train)

        pred = selected_model.predict(xv_test)

        # Input Text for detection
        news = st.text_area("Masukkan teks berita:")

        # Detect button
        if st.button("Deteksi"):
            def manual_testing(news):
                testing_news = {"text": [news]}
                new_def_test = pd.DataFrame(testing_news)
                new_def_test["text"] = new_def_test["text"].apply(wordopt)
                new_x_test = new_def_test["text"]
                new_xv_test = vectorization.transform(new_x_test)
                pred = selected_model.predict(new_xv_test)
                return "Berita Hoaks" if pred[0] == 0 else "Berita Real"

            result = manual_testing(news)
            st.write(f"Hasil Deteksi: {result}")

    elif menu == "Visualisasi":
        st.header("Visualisasi Data")

        # Input Text for visualization
        vis_news = st.text_area("Masukkan teks berita untuk visualisasi:")

        # Check if input is provided
        if not vis_news:
            st.warning("Harap masukkan teks berita terlebih dahulu untuk melihat visualisasi.")
        else:
            if st.sidebar.checkbox("Tampilkan Confusion Matrix"):
                model_choice = st.sidebar.selectbox("Pilih model", list(models.keys()))

                selected_model = models[model_choice]
                selected_model.fit(xv_train, y_train)
                pred = selected_model.predict(xv_test)

                st.write(f"Confusion Matrix untuk model {model_choice}:")
                cm = confusion_matrix(y_test, pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

            if st.sidebar.checkbox("Tampilkan Word Cloud"):
                st.write("Word Cloud")

                fake_text = " ".join(df_fake["text"].astype(str).tolist())
                true_text = " ".join(df_true["text"].astype(str).tolist())

                wordcloud_fake = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
                wordcloud_true = WordCloud(width=800, height=400, background_color='white').generate(true_text)

                st.write("### Word Cloud untuk Berita Hoaks")
                fig_fake, ax_fake = plt.subplots()
                ax_fake.imshow(wordcloud_fake, interpolation='bilinear')
                ax_fake.axis('off')
                st.pyplot(fig_fake)

                st.write("### Word Cloud untuk Berita Real")
                fig_true, ax_true = plt.subplots()
                ax_true.imshow(wordcloud_true, interpolation='bilinear')
                ax_true.axis('off')
                st.pyplot(fig_true)

        # Add button to go back to model selection
        if st.sidebar.button("Kembali ke Pemilihan Model"):
            st.session_state.menu = "Pemilihan Model"

    elif menu == "EDA":
        st.header("Exploratory Data Analysis")

        # Check if input is provided
        if not x_test.shape[0]:
            st.warning("Harap lakukan pemisahan data terlebih dahulu sebelum menjalankan EDA.")
        else:
            st.write("### Statistik Dasar")
            st.write(df.describe())

            st.write("### Distribusi Kelas Berita")
            fig, ax = plt.subplots()
            sns.countplot(x="class", data=df, ax=ax)
            st.pyplot(fig)

            st.write("### Distribusi Panjang Teks")
            df['text_length'] = df['text'].apply(len)
            fig, ax = plt.subplots()
            sns.histplot(df['text_length'], bins=30, ax=ax)
            st.pyplot(fig)

        # Add button to go back to model selection
        if st.sidebar.button("Kembali ke Pemilihan Model"):
            st.session_state.menu = "Pemilihan Model"

    elif menu == "Perbandingan Model":
        st.header("Perbandingan Model")

        # Check if input is provided
        if not x_test.shape[0]:
            st.warning("Harap lakukan pemisahan data terlebih dahulu sebelum melakukan perbandingan model.")
        else:
            results = {}
            for name, model in models.items():
                model.fit(xv_train, y_train)
                pred = model.predict(xv_test)
                results[name] = {
                    "accuracy": accuracy_score(y_test, pred),
                    "precision": classification_report(y_test, pred, output_dict=True)['weighted avg']['precision'],
                    "recall": classification_report(y_test, pred, output_dict=True)['weighted avg']['recall'],
                    "f1-score": classification_report(y_test, pred, output_dict=True)['weighted avg']['f1-score']
                }

            results_df = pd.DataFrame(results).T
            st.write(results_df)

            st.write("### Perbandingan Performa Model")
            fig, ax = plt.subplots()
            results_df.plot(kind='bar', ax=ax)
            st.pyplot(fig)

        # Add button to go back to model selection
        if st.sidebar.button("Kembali ke Pemilihan Model"):
            st.session_state.menu = "Pemilihan Model"

    # Logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False

# Login form
else:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")
