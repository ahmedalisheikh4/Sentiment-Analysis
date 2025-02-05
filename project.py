import json
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, GlobalMaxPooling1D, Dropout, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tkinter import scrolledtext, messagebox

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

# Read JSON files from the review and abstract folders
review_folder_path = r'ICLR_2017_review/'
abstract_folder_path = r'ICLR_2017_content/'
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

# class weights as a dictionary
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
print(class_weights)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# MILAM model with attention mechanism
def create_milam_model_with_attention(num_classes):
    input_review = Input(shape=(300,), dtype='int32')

    embedding_layer = Embedding(10000, 300)

    review_embedding = embedding_layer(input_review)

    review_lstm = Bidirectional(LSTM(128, return_sequences=True))(review_embedding)
    review_lstm = Dropout(0.5)(review_lstm)  # dropout layer

    # Attention mechanism
    review_attention = Attention()([review_lstm, review_lstm])

    #feature selection
    pooled = GlobalMaxPooling1D()(review_attention)

    dense_layer = Dense(256, activation='tanh')(pooled)
    dense_layer = Dropout(0.5)(dense_layer)
    #number of classes  
    output = Dense(num_classes, activation='softmax')(dense_layer)

    model = Model(inputs=input_review, outputs=output)
    
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #ross-entropy is a loss function that measures the difference between two probability distributions: the predicted probability distribution output by the model and the true probability distribution of the labels.
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


num_classes = 3  # We have three classes: accept, reject and borderline
model = create_milam_model_with_attention(num_classes)
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=256, validation_data=(X_test, y_test), class_weight=class_weights_dict)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()


# Function to predict and display sentiment
def predict_sentiment(review_text):
    print(review_text)
    review_seq = tokenizer.texts_to_sequences([review_text])    
    review_pad = pad_sequences(review_seq, maxlen=300)
    
    prediction = model.predict(review_pad)
    # Make predictions on the test set
    y_pred = model.predict(X_test).argmax(axis=1)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    sentiment = 'accept' if prediction.argmax(axis=1)[0] == 1 else 'reject' if prediction.argmax(axis=1)[0] == 0 else 'borderline'
    
    print("Prediction:", prediction)
    print("Sentiment:", sentiment)
    return sentiment


class SentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Peer Review Sentiment Analysis")
        
        self.lbl_review = tk.Label(root, text="Review Text")
        self.lbl_review.pack()
        
        self.txt_review = scrolledtext.ScrolledText(root, width=50, height=10)
        self.txt_review.pack()
        
        self.btn_analyze = tk.Button(root, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.btn_analyze.pack()
        
        self.lbl_result = tk.Label(root, text="Sentiment Result")
        self.lbl_result.pack()
        
        self.txt_result = scrolledtext.ScrolledText(root, width=50, height=10)
        self.txt_result.pack()
    
    def analyze_sentiment(self):
        review_text = self.txt_review.get("1.0", tk.END).strip()
        
        if not review_text:
            messagebox.showwarning("Input Error", "Please enter review text")
            return
        
        sentiment = predict_sentiment(review_text)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, sentiment)

# main window for tkinter
root = tk.Tk()
app = SentimentAnalysisApp(root)
root.mainloop()
