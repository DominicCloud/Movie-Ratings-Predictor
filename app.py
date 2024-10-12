# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        # Calculate average rating (scaled down by factor of 10 for precision  upto 2 decimals places)
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

# Run the Flask app if this script is executed
if __name__ == '__main__':
    app.run(debug=True)