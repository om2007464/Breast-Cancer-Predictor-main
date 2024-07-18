# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Function to clean and preprocess the data
def get_clean_data():
    # Load the dataset
    data = pd.read_csv("data/data.csv")
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Map diagnosis values to numerical values
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data

# Function to create and train the model
def create_model(data):
    # Split data into features (X) and target (y)
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test the model and print accuracy and classification report
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))
    
    return model, scaler

# Main function to execute the workflow
def main():
    # Clean and preprocess the data
    data = get_clean_data()
    
    # Create and train the model
    model, scaler = create_model(data)
    
    # Save the trained model to a file
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save the scaler to a file
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Entry point of the script
if __name__ == '__main__':
    main()
