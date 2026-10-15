# Titanic Survival Prediction

A beginner-friendly Machine Learning mini project using the Kaggle Titanic dataset.

## Technologies
- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Streamlit
- Matplotlib

## How to Run in VS Code

Open the project folder in VS Code and open Terminal.

### 1. Install libraries
```powershell
pip install -r requirements.txt
```

### 2. Train the model
```powershell
python model.py
```

This creates `titanic_model.pkl` and prints model accuracy.

### 3. Start the website
```powershell
streamlit run app.py
```

If `streamlit` is not recognized:
```powershell
python -m streamlit run app.py
```

## Prediction
Enter passenger details and click **Predict Survival**.

Output:
- Passenger is predicted to SURVIVE
- Passenger is predicted NOT TO SURVIVE

## Dataset
The included `train.csv` is the Titanic training dataset supplied for this project.
