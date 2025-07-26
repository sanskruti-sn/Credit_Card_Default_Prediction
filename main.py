from src.data_ingestion import load_data
from src.preprocessing import preprocess_data
from src.model import train_model
import pickle
from sklearn.model_selection import train_test_split

def main():
    df = load_data()
    X, y, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train, X_test, y_test)

    # Save scaler and model
    with open('artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    with open('artifacts/model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()
