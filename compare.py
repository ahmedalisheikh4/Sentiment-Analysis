import json
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Attention
from sklearn.linear_model import LogisticRegression

# Function to read JSON files from a folder and extract review data
def read_reviews(folder_path):
    reviews = []
    positive = ['good' , 'strong', 'accept' , 'strongly accept','good paper','above']
    borderline = ['marginally', 'probably', 'maybe',  'neutral']
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, encoding="utf8") as f:
                data = json.load(f)
            paper_id = data['id']
            for review in data['reviews']:
                review_text = review['review']
                rating = review['rating']
                confidence = review['confidence']
                if any(pos in rating.lower() for pos in positive):
                    preliminary_decision = 1
                elif any(border in rating.lower() for border in borderline):
                    preliminary_decision = 2
                else:
                    preliminary_decision = 0
                reviews.append({
                    'paper_id': paper_id,
                    'text': review_text,
                    'rating': rating,
                    'confidence': confidence,
                    'preliminary_decision': preliminary_decision
                })
    return reviews

# Read JSON files from the review folder
review_folder_path = r'ICLR_2017_review/'
review_papers = read_reviews(review_folder_path)

# Convert to DataFrame
df = pd.DataFrame(review_papers)

# Remove rows with None in 'text' column
df = df.dropna(subset=['text'])

# Preprocess the text data
tokenizer = Tokenizer(num_words=10000)
texts = df['text'].tolist()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=300)

# Encoding labels
filtered_df = df.dropna(subset=['preliminary_decision'])
labels = filtered_df['preliminary_decision'].astype(int).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences[:len(filtered_df)], labels, test_size=0.2, random_state=42)

# Class weights as a dictionary
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Helper function to plot the training history
def plot_training_history(history, model_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

# SVM with unigrams, bigrams, and sentiment features
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_tfidf = tfidf.fit_transform(filtered_df['text'].tolist())

# Train-test split for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Train SVM model
svm = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', class_weight='balanced'))
svm.fit(X_train_svm, y_train_svm)

# Evaluate SVM model
svm_val_accuracy = svm.score(X_test_svm, y_test_svm)
print(f'SVM Validation Accuracy: {svm_val_accuracy}')


# SVM with unigrams, bigrams, and sentiment features
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_tfidf = tfidf.fit_transform(filtered_df['text'].tolist())

# Train-test split for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Train SVM model
svm = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', class_weight='balanced'))
svm.fit(X_train_svm, y_train_svm)

# Evaluate SVM model
svm_val_accuracy = svm.score(X_test_svm, y_test_svm)
print(f'SVM Validation Accuracy: {svm_val_accuracy}')


# Simple LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=300))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model()
history_lstm = lstm_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights_dict)
lstm_val_accuracy = history_lstm.history['val_accuracy'][-1]
print(f'LSTM Validation Accuracy: {lstm_val_accuracy}')


# CNN + Bi-LSTM + Attention model
def create_cnn_bilstm_attention_model():
    input_review = Input(shape=(300,), dtype='int32')
    embedding_layer = Embedding(10000, 128)(input_review)
    
    conv_layer = Conv1D(128, 5, activation='relu')(embedding_layer)
    pooling_layer = MaxPooling1D(5)(conv_layer)
    
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(pooling_layer)
    attention_layer = Attention()([lstm_layer, lstm_layer])
    
    global_pooling_layer = GlobalMaxPooling1D()(attention_layer)
    dense_layer = Dense(128, activation='relu')(global_pooling_layer)
    output = Dense(3, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_review, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cnn_bilstm_attention_model = create_cnn_bilstm_attention_model()
history_cnn_bilstm_attention = cnn_bilstm_attention_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights_dict)
cnn_bilstm_attention_val_accuracy = history_cnn_bilstm_attention.history['val_accuracy'][-1]
print(f'CNN + Bi-LSTM + Attention Validation Accuracy: {cnn_bilstm_attention_val_accuracy}')


# MIL model
def create_mil_model():
    input_review = Input(shape=(300,), dtype='int32')
    embedding_layer = Embedding(10000, 128)(input_review)
    lstm_layer = LSTM(128, return_sequences=True)(embedding_layer)
    global_pooling_layer = GlobalMaxPooling1D()(lstm_layer)
    dense_layer = Dense(128, activation='relu')(global_pooling_layer)
    output = Dense(3, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_review, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

mil_model = create_mil_model()
history_mil = mil_model.fit(X_train, y_train, epochs=10, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights_dict)
mil_val_accuracy = history_mil.history['val_accuracy'][-1]
print(f'MIL Validation Accuracy: {mil_val_accuracy}')


# MILAM model with attention mechanism
def create_milam_model_with_attention(num_classes):
    input_review = Input(shape=(300,), dtype='int32')
    embedding_layer = Embedding(10000, 300)(input_review)
    review_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    review_lstm = Dropout(0.5)(review_lstm)
    review_attention = Attention()([review_lstm, review_lstm])
    pooled = GlobalMaxPooling1D()(review_attention)
    dense_layer = Dense(256, activation='tanh')(pooled)
    dense_layer = Dropout(0.5)(dense_layer)
    output = Dense(num_classes, activation='softmax')(dense_layer)
    model = Model(inputs=input_review, outputs=output)
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

num_classes = 3
milam_model_with_attention = create_milam_model_with_attention(num_classes)
history_milam_with_attention = milam_model_with_attention.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights_dict)
milam_with_attention_val_accuracy = history_milam_with_attention.history['val_accuracy'][-1]
print(f'MILAM with Attention Validation Accuracy: {milam_with_attention_val_accuracy}')


# Validation accuracies of different models
model_names = ['SVM', 'LSTM', 'CNN + Bi-LSTM + Attention', 'MIL', 'MILAM with Attention']
val_accuracies = [svm_val_accuracy, lstm_val_accuracy, cnn_bilstm_attention_val_accuracy, mil_val_accuracy, milam_with_attention_val_accuracy]

# Plotting validation accuracies
plt.figure(figsize=(10, 5))
plt.bar(model_names, val_accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Model')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy Comparison of Different Models')
plt.show()
