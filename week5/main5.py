import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import zipfile
import requests
import io

# Load the dataset
@st.cache
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('SMSSpamCollection') as f:
            df = pd.read_csv(f, sep='\t', names=['label', 'message'])
    return df

df = load_data()

# Preprocess the data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize the text data
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Train the Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Make predictions
y_pred = clf.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Streamlit app
st.title('Naive Bayes Document Classification')

st.write('## Dataset')
st.write(df.head())

st.write('## Model Performance')
st.write(f'Accuracy: {accuracy:.4f}')
st.write(f'Precision: {precision:.4f}')
st.write(f'Recall: {recall:.4f}')

st.write('## Sample Predictions')
num_samples = st.slider('Number of samples to display', min_value=1, max_value=10, value=5)
sample_indices = list(range(len(X_test)))[:num_samples]

for i in sample_indices:
    st.write(f'**Message:** {X_test.iloc[i]}')
    st.write(f'**Actual Label:** {"spam" if y_test.iloc[i] == 1 else "ham"}')
    st.write(f'**Predicted Label:** {"spam" if y_pred[i] == 1 else "ham"}')
    st.write('---')
