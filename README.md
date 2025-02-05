# 🔍 Sentiment Analysis of Peer Reviews

## 📝 Overview
This project predicts the sentiment of peer reviews from the **ICLR 2017 dataset** into three categories:
- ✅ **Accept**
- ❌ **Reject**
- ⚠️ **Borderline**

It combines **LSTM with Attention** (Deep Learning) and **SVM with TF-IDF** (Machine Learning) to classify reviews.

## 🚀 Features
- 📖 Predicts sentiment of academic reviews.
- 🤖 Uses Deep Learning (LSTM with Attention) and Machine Learning (SVM with TF-IDF).
- 📊 Evaluates performance using accuracy and confusion matrix.
- 🎨 Includes a Tkinter-based GUI for easy interaction.

## 🛠 Technologies Used
- **Flask** (Web Framework)
- **Pandas, NumPy** (Data Processing)
- **Scikit-learn** (SVM Model)
- **TensorFlow/Keras** (LSTM Model)
- **Matplotlib** (Visualization)
- **Tkinter** (GUI)

## 📥 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-analysis-peer-reviews.git
cd sentiment-analysis-peer-reviews

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```


