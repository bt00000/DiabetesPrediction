from django.shortcuts import render
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
from optuna.integration import TFKerasPruningCallback
from django.conf import settings  # For dynamic paths

# File paths for datasets
train_file_path = os.path.join(settings.BASE_DIR, 'DiabetesPrediction', 'datasets', 'diabetes75pc_100_times.csv')
test_file_path = os.path.join(settings.BASE_DIR, 'DiabetesPrediction', 'datasets', 'diabetes25pc.csv')

# Load data
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Combine train and test for splitting
combined_data = pd.concat([train_data, test_data]).sample(frac=1, random_state=42)

# Selected features based on analysis
selected_features = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction']
X = combined_data[selected_features]
y = combined_data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optuna optimization
def objective(trial):
    model = Sequential()
    # Add layers with units suggested by Optuna
    for i in range(3):
        units = trial.suggest_int(f"units_layer_{i}", 16, 128, step=8)
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(0.2))  # Dropout for regularization

    model.add(Dense(1, activation="sigmoid"))  # Output layer

    # Learning rate and batch size
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3),
        TFKerasPruningCallback(trial, "val_accuracy"),
    ]

    # Train
    model.fit(X_train, y_train, validation_split=0.2, epochs=50,
              batch_size=trial.suggest_int("batch_size", 32, 128, step=16),
              callbacks=callbacks, verbose=0)

    val_loss, val_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return val_accuracy

# Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# Get the best trial
best_trial = study.best_trial
print("Best trial:", best_trial.params)

# Build the final model
final_model = Sequential()
for i in range(3):
    units = best_trial.params[f"units_layer_{i}"]
    final_model.add(Dense(units, activation="relu"))
    final_model.add(Dropout(0.2))

final_model.add(Dense(1, activation="sigmoid"))
final_model.compile(optimizer=Adam(learning_rate=best_trial.params["learning_rate"]),
                    loss="binary_crossentropy", metrics=["accuracy"])

# Train the final model
final_model.fit(X_train, y_train, epochs=50, batch_size=best_trial.params["batch_size"], verbose=1)

# Home view
def home(request):
    return render(request, 'home.html')

# Predict view
def predict(request):
    result = None

    # Calculate accuracy on the test set
    y_pred = (final_model.predict(X_test) > 0.5).astype(int).flatten()
    accuracy = (y_pred == y_test).mean() * 100

    if request.method == "GET" and 'n1' in request.GET:
        try:
            user_input = [
                float(request.GET.get('n1')),  # Pregnancies
                float(request.GET.get('n2')),  # Glucose
                float(request.GET.get('n3')),  # BMI
                float(request.GET.get('n4')),  # DiabetesPedigreeFunction
            ]
            scaled_input = scaler.transform([user_input])
            prediction = (final_model.predict(scaled_input) > 0.5).astype(int)[0][0]
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        except Exception as e:
            result = f"Error: {e}"

    return render(request, 'predict.html', {
        'result2': result,
        'accuracy': accuracy,
    })
