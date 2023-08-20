import pandas as pd
import pickle

def load_pickle_model(file_path):
    # Load the model from the file path specified
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    return model

def preprocess_data(df):
    # Drop the deposit column
    X = df.drop("deposit", axis=1)

    # Specify the target variable
    y = df["deposit"]

    # Specify the categorical and numerical columns
    cat_cols = ["job", "marital", "education", "housing"]
    num_cols = ["age", "balance", "duration", "campaign"]

    # Load the OneHotEncoder
    one_hot_enc = load_pickle_model("../models/ohe.pkl")

    # Transform the data
    X = pd.DataFrame(one_hot_enc.transform(X), columns=one_hot_enc.get_feature_names_out())

    return X

def predict(model, data):
    predictions = model.predict(data)
    
    # Map predictions to 'yes' or 'no'
    mapped_predictions = ["yes" if pred == 1 else "no" for pred in predictions]
    
    return mapped_predictions

def main():
    # Load the model
    model = load_pickle_model("../models/model.pkl")

    # Load the OneHotEncoder
    one_hot_enc = load_pickle_model("../models/ohe.pkl")

    # Load the data
    df = pd.read_csv("../data/bank_processed.csv")

    # Preprocess the data
    X = preprocess_data(df)

    # Make predictions
    y_pred = predict(model, X)

    # Add the predictions to the dataframe
    df['y_pred'] = y_pred

    # Save the predictions back to the CSV file if needed
    df.to_csv("../data/bank_processed_with_predictions.csv", index=False)

    # Print the predictions
    print(y_pred)

if __name__ == "__main__":
    main()
