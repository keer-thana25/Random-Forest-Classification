# Titanic Survival Prediction using Random Forest Classification

## Project Overview

This project is an interactive **Streamlit web application** that predicts whether a passenger is likely to survive on the Titanic using the **Random Forest Classification algorithm**.

The application provides an attractive and interactive interface where users can:

- Understand what Random Forest is
- Explore the Titanic dataset
- Visualize and handle missing values
- Detect and remove outliers
- Perform preprocessing techniques
- Train a Random Forest model
- Predict passenger survival using custom inputs

---

## Live Demo

🔗 **Live Application:**  
Coming Soon

Replace this with your Streamlit deployment link:

https://your-app-name.streamlit.app

---

## What is Random Forest?

Random Forest is a Machine Learning algorithm used for **classification and prediction**.

It works by creating **multiple decision trees** and combining their predictions to produce the final output.

### Simple Explanation

Imagine asking multiple people for an answer.

- Person 1 says: Survive  
- Person 2 says: Not Survive  
- Person 3 says: Survive  

The final decision is based on the **majority vote**.

This makes Random Forest more accurate and reliable than a single decision tree.

### Why Random Forest?

Random Forest is used because:

- It provides high accuracy
- It reduces overfitting
- It handles large datasets efficiently
- It works well with missing data
- It gives reliable predictions

---

## Dataset Used

This project uses the **Titanic Dataset**, which contains passenger details to predict survival.

The dataset includes passenger information such as:

- Passenger Class
- Gender
- Age
- Fare
- Family Information
- Boarding Location

---

## Dataset Columns Explanation

### PassengerId
Unique passenger identification number.

### Survived
Target column.

- `0` → Did Not Survive  
- `1` → Survived

### Pclass
Passenger travel class.

- `1` → First Class  
- `2` → Second Class  
- `3` → Third Class

### Name
Passenger name.

### Sex
Passenger gender.

- Male  
- Female

### Age
Passenger age.

### SibSp
Number of siblings/spouses traveling together.

### Parch
Number of parents/children traveling together.

### Ticket
Ticket number.

### Fare
Amount paid for the ticket.

### Cabin
Cabin number.

### Embarked
Passenger boarding location.

- `C` → Cherbourg  
- `Q` → Queenstown  
- `S` → Southampton

---

## Features of the Application

### Random Forest Explanation
A beginner-friendly explanation of Random Forest with simple examples.

### Titanic Dataset Exploration
- Dataset head  
- Dataset shape  
- Column names  
- Dataset information  
- Statistical summary  

### Missing Value Handling
- Missing values before handling  
- Missing values after handling  

### Outlier Detection
- Outliers before handling  
- Outliers after handling  
- Boxplot visualization for numeric columns  

### Data Preprocessing
- Removing unnecessary columns  
- Encoding categorical variables  
- Feature scaling using StandardScaler  
- Train-test split  

### Model Training

The Random Forest model is trained using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Model Evaluation

- Accuracy Score  
- Confusion Matrix  
- Classification Report  

### Feature Importance

Visual representation of important features affecting survival.

### Decision Tree Visualization

Displays one decision tree from the Random Forest.

### Interactive Prediction

Users can enter passenger details and predict whether the passenger is likely to survive.

---

## Technologies Used

- Python  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-Learn  

---

## Installation

Clone the repository:

```bash
git clone (https://github.com/keer-thana25/Random-Forest-Classification.git)
```

Move to the project folder:

```bash
cd your-repository-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run the Streamlit application using:

```bash
streamlit run app.py
```

---

## Project Workflow

1. Load Titanic Dataset  
2. Handle Missing Values  
3. Detect Outliers  
4. Remove Outliers  
5. Data Visualization  
6. Feature Engineering  
7. Feature Scaling  
8. Train-Test Split  
9. Random Forest Model Training  
10. Model Evaluation  
11. Passenger Survival Prediction  

---

## Author

**Keerthana**
