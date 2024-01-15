import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset without assuming the first row as headers
file_path = r'C:\Users\Demirkaya\Desktop\ANN Final\myData.xlsx'  # Replace with the actual path to your Excel file
df = pd.read_excel(file_path, header=None)

# Assign original column names
df.columns = df.iloc[0]

# Drop the first row (which contains the original column names) after assigning them
df = df.drop(0)

# Impute missing values with mean for numerical columns
imputer = SimpleImputer(strategy='mean')
df[df.columns[1:4]] = imputer.fit_transform(df[df.columns[1:4]])

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Convert categorical variables to numerical representation using Label Encoding
le = LabelEncoder()
df[df.columns[4]] = le.fit_transform(df[df.columns[4]])

# Split the dataset into features (X) and target variable (y)
X = df[df.columns[1:4]]
y = df[df.columns[4]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of the preprocessed data
print("Preprocessed Data:")
print(df)

# Define the neural network model
model = Sequential()
model.add(Dense(64, input_dim=3, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions on the test set
predictions = model.predict(X_test)

# Convert the predictions to binary labels
binary_predictions = (predictions > 0.5).astype(int)

# Calculate accuracy on the test set
accuracy_test = accuracy_score(y_test, binary_predictions)
print(f'Accuracy on Test Set: {accuracy_test * 100:.2f}%')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the predicted outcomes
ax.scatter(range(len(y_test)), binary_predictions, color='blue', label='Predicted', alpha=0.7)

# Plot the true outcomes
ax.scatter(range(len(y_test)), y_test, color='red', label='True', alpha=0.7)

# Set plot labels and title
ax.set_xlabel('Data Points')
ax.set_ylabel('Final State (0 or 1)')
ax.set_title('Predicted vs True Outcomes for Each Coin Drop')

# Show legend
ax.legend()

# Show the plot
plt.show()