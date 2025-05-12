# ROCK-VS-MINE-PREDICTION-USING-SONAR-DATA
# Rock vs Mine Classification

This project uses machine learning to classify sonar signals as either **rocks** or **mines**. The dataset contains 60 features representing sonar signal strengths bounced off different objects. The model is trained using supervised learning and can make real-time predictions for new sonar inputs.

---

## 🔍 Problem Statement

Build a binary classification model that predicts whether an object detected by sonar is:

- **Rock (R)** – Natural underwater formations
- **Mine (M)** – Potentially dangerous metallic objects

---

## 🛠️ Tools & Libraries

- Python 3
- NumPy
- Pandas
- scikit-learn
- Google Colab

---

## ✅ Features

- Data exploration and visualization
- Preprocessing and standardization
- Model training and evaluation
- Prediction on custom input
- Easy to use in Google Colab

---

## 📊 Example Prediction Code

```python
input_data = (0.0239,0.0189,0.0466,0.0440,0.0657,0.0742,0.1380,0.1099,0.1384,0.1376,0.0938,0.0259,0.1499,0.2851,0.5743,0.8278,0.8669,0.8131,0.9045,0.9046,1.0000,0.9976,0.9872,0.9761,0.9009,0.9724,0.9675,0.7633,0.4434,0.3822,0.4727,0.4007,0.3381,0.3172,0.2222,0.0733,0.2692,0.1888,0.0712,0.1062,0.0694,0.0300,0.0893,0.1459,0.1348,0.0391,0.0546,0.0469,0.0201,0.0095,0.0155,0.0091,0.0151,0.0080,0.0018,0.0078,0.0045,0.0026,0.0036,0.0024)
#changing the data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the np array as we predicting from one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0]=='R'):
    print('The object is a rock')
else:
    print('The object is a mine')

```
**##📈 Model Used**
Model Type: LR 
Test Accuracy: 0.76
