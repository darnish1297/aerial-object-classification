# aerial-object-classification
Deep learning-based aerial object classifier using ResNet50 to distinguish between drones and birds. Includes data preprocessing, model training, and Streamlit web app for real-time prediction.


## 🚀 Features
- Data Augmentation using Keras ImageDataGenerator  
- Transfer Learning with ResNet50  
- Streamlit Web App for real-time image classification  

## 🧠 Model
- Fine-tuned on custom dataset  
- Test Accuracy: **83.72%**  
- Test Loss: **0.386**

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
