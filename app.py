from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load the dataset and train the model
df = pd.read_csv('movie_ratings.csv')

# Define features and target
X = df[['Gender', 'Age', 'Genre', 'Previous Ratings']]
y = df['Predicted Rating']

# Preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Gender', 'Genre']),
    ],
    remainder='passthrough'  # Keep numerical features
)

# Create the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_value = None
    if request.method == 'POST':
        gender = request.form['gender']
        age = int(request.form['age'])
        genre = request.form['genre']
        previous_rating = float(request.form['previous_rating'])

        # Make prediction
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Genre': [genre],
            'Previous Ratings': [previous_rating]
        })
        predicted_value = model.predict(input_data)[0]

    return render_template('index.html', predicted_value=predicted_value)

if __name__ == '__main__':
    app.run(debug=True)
