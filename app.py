# Import necessary libraries
from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import os

# Initialize Flask application
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('movie_ratings.csv')

# Define features (X) and target (y) for the model
X = df[['Gender', 'Age', 'Genre', 'Previous Ratings']]
y = df['Predicted Rating']

# Create a preprocessor for handling categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Gender', 'Genre']),  # OneHotEncode categorical features
    ],
    remainder='passthrough'  # Keep numerical features as is
)

# Create a pipeline that combines preprocessing and the linear regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Log performance measures to the console
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Ensure the static directory exists
static_dir = 'static'
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Plot the performance metrics and save as images
def plot_metrics():
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    names = list(metrics.keys())
    values = list(metrics.values())

    # Bar Chart of Performance Metrics
    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color=['#6e44ff', '#ffe600', '#f06292'])
    plt.title('Performance Metrics (Bar Chart)')
    plt.ylabel('Value')
    plt.xlabel('Metric')
    plt.savefig(os.path.join(static_dir, 'performance_metrics_bar.png'))
    plt.close()

    # Histogram of residuals (errors)
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30, color='cyan')
    plt.title('Residuals Histogram')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(static_dir, 'residuals_histogram.png'))
    plt.close()

    # Scatterplot of true vs predicted values
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='orange')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.title('True vs Predicted Values (Scatterplot)')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.savefig(os.path.join(static_dir, 'true_vs_pred_scatter.png'))
    plt.close()

# Generate and save the graphs when the app starts
plot_metrics()

# Define route for both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def movies():
    average_rating = None
    predicted_value = None

    if request.method == 'POST':
        # Collect movie ratings from the form
        ratings = [
            float(request.form['movie1']),
            float(request.form['movie2']),
            float(request.form['movie3']),
            float(request.form['movie4']),
            float(request.form['movie5'])
        ]
        # Calculate average rating (scaled down by factor of 10 for precision up to 2 decimals places)
        average_rating = sum(ratings) / (10 * len(ratings))

        # Collect other user inputs
        gender = request.form['gender']
        age = int(request.form['age'])
        genre = request.form['genre']

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Genre': [genre],
            'Previous Ratings': [average_rating]
        })

        # Make prediction
        predicted_value = model.predict(input_data)[0]

    # Render the template with results
    return render_template('movies.html', predicted_value=predicted_value, average_rating=average_rating)

# Define a new route to display graphs
@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

# Run the Flask app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
