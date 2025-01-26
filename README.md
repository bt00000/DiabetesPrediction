![diabetes-prediction](https://github.com/user-attachments/assets/85ad22ef-3092-46bd-8693-727004b1dc74)

# DiabetesPredictionApp

## Overview
DiabetesPredictionApp is a Django-based web application designed to predict diabetes outcomes using machine learning. The application achieves **91-98% accuracy** by leveraging TensorFlow for modeling, Optuna for hyperparameter optimization, and effective feature scaling with StandardScaler. The app provides an interactive interface for users to input health data and receive predictions.

---

## Key Features
- **High Accuracy:** Achieves 91-98% prediction accuracy on test datasets.
- **Hyperparameter Optimization:** Uses Optuna to tune model parameters like hidden layers, learning rate, and batch size.
- **Interactive Web Interface:** Accepts user inputs for eight health metrics and provides diabetes predictions in real-time.
- **Data Handling:** Processes pre-split training and testing datasets, dynamically loaded from local directories.
- **Scalable Machine Learning Pipeline:** Implements a fully connected neural network optimized for binary classification.

---

## Technical Details
### Technologies Used
- **Backend:** Django
- **Machine Learning:** TensorFlow, Keras
- **Hyperparameter Optimization:** Optuna
- **Data Preprocessing:** StandardScaler (scikit-learn)

### Workflow
1. **Datasets:**
   - `diabetes75pc_100_times.csv` (training)
   - `diabetes25pc.csv` (testing)
2. **Features:**  
   `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
3. **Model:**  
   Sequential Neural Network with three hidden layers.
4. **Training:**  
   Uses validation split with early stopping and pruning callbacks.
5. **Testing:**  
   Measures accuracy using a separate test dataset.

### Prediction Process
1. Users input health data via the web interface.
2. Data is scaled and passed through the trained model.
3. The app displays whether the user is predicted to be diabetic or non-diabetic.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DiabetesPredictionApp.git
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
3. Start the Django server:
   ```bash
   python manage.py runserver

## Results

### Training and Prediction Accuracy
- **Training Accuracy:** 91-98%
- **Prediction Accuracy (on test dataset):** 91-98%

## Future Enhancements
- Add visualization for model performance metrics such as ROC curves or confusion matrices.
- Extend support for additional health datasets and predictive metrics.
- Implement multi-class classification for extended healthcare diagnostics.

   
