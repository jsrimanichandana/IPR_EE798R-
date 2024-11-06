import numpy as np
import pandas as pd
import pyedflib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Set the path to the downloaded data directory
data_dir = 'physionet.org/files/eegmat/1.0.0'  # Update the path if necessary

# Fixed sequence length for padding or truncation
FIXED_SEQUENCE_LENGTH = 6000  # Adjust based on your dataset characteristics

# Function to load EEG data from .edf files
def load_eeg_data(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signals = np.zeros((n, f.getNSamples()[0]))
    for i in range(n):
        signals[i, :] = f.readSignal(i)
    f.close()
    return signals

# Placeholder for DWT (Discrete Wavelet Transform)
def perform_dwt(signals):
    # Add actual DWT processing here if necessary
    return signals  # Placeholder, assuming signals are processed

# Preprocess and extract features from EEG signals
def preprocess_and_extract_features(signals):
    denoised_signals = perform_dwt(signals)
    return denoised_signals  # return processed signals for CNN input

# Pad or truncate signals to a fixed length
def pad_or_truncate(signals, length):
    if signals.shape[1] > length:
        return signals[:, :length]  # Truncate if longer
    else:
        # Pad with zeros if shorter
        padded = np.zeros((signals.shape[0], length))
        padded[:, :signals.shape[1]] = signals
        return padded

# Define CNN-BLSTM model with more layers
def create_cnn_blstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess data
subject_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.edf')]
X, y = [], []
for file_path in subject_files:
    signals = load_eeg_data(file_path)
    signals = preprocess_and_extract_features(signals)
    signals = pad_or_truncate(signals, FIXED_SEQUENCE_LENGTH)  # Pad or truncate each signal
    X.append(signals)
    # Label each file as stress (1) or relax (0) based on _1 or _2 in the filename
    y.append(1 if '_2' in file_path else 0)
X, y = np.array(X), np.array(y)

# Reshape X to have a consistent input shape (num_samples, sequence_length, num_channels)
X = X.reshape((X.shape[0], FIXED_SEQUENCE_LENGTH, -1))

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Train model with reduced learning rate
model = create_cnn_blstm_model(X_train.shape[1:])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Confusion Matrix and Additional Metrics
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
npv = tn / (tn + fn)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
positive_likelihood_ratio = sensitivity / (1 - specificity)
negative_likelihood_ratio = (1 - sensitivity) / specificity

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"NPV: {npv:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"+LR: {positive_likelihood_ratio:.4f}")
print(f"-LR: {negative_likelihood_ratio:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot training and validation accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
