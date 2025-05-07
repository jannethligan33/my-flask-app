# Import necessary libraries 
import pandas as pd  # For data handling 
import pickle       # For saving the trained model 
from sklearn.model_selection import train_test_split  # For splitting data 
from sklearn.linear_model import LinearRegression  # For fitting the model
from flask import Flask, request, jsonify 

# Load the dataset from a CSV file 
df = pd.read_csv('Ecommerce_Customers.csv')

# Define the features (input) and label (output) columns 
features = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership'] 
label = "Yearly Amount Spent"

# Extract input features (X) and output labels (y) 
X = df[features] 
y = df[label]

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create a Linear Regression model 
regression_model = LinearRegression()

# Train the model on the training data 
regression_model.fit(X_train, y_train)

# Make predictions using the trained model 
predictions = regression_model.predict(X_test)

# Print the model's predictions 
print(predictions)


# Save the trained model to a file named "model.pkl" 
pickle.dump(regression_model, open("model.pkl", "wb"))


# Create a Flask app 
app = Flask(__name__)


# Load the machine learning model from a pickle file 
model = pickle.load(open("model.pkl", "rb"))

@app.route('/keepalive', methods=['GET']) 
def api_health(): 
    return jsonify(Message="Success")


# Define a route for making predictions 
@app.route("/predict", methods=["POST"]) 
def predict(): 
    # Get JSON data from the request 
    json_ = request.json 
 
    # Convert JSON data into a DataFrame 
    df = pd.DataFrame(json_) 
 
    # Use the loaded model to make predictions on the DataFrame 
    prediction = model.predict(df) 
 
    # Return the predictions as a JSON response 
    return jsonify({"Prediction": list(prediction)}) 
# Run the Flask app when this script is executed 
if __name__ == "__main__": 
    app.run(debug=True)
    
    