# Diabetector-Care
Diabetes Prediction using Random Forest  A machine learning model that predicts diabetes based on health features like glucose levels and BMI. The project includes data preprocessing, model training with Random Forest, and a real-time prediction tool for assessing diabetes risk.

# Diabetes Prediction Using Random Forest Classifier

## Overview

This project aims to predict whether an individual is diabetic or not based on health-related features using a **Random Forest Classifier**. The model is trained on a dataset containing various medical attributes such as glucose level, BMI, insulin, and age. The project also includes a real-time prediction tool where users can input personal health metrics and receive a diabetes risk prediction.

## Technologies Used
- **Python** (3.6+)
- **Libraries:**
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `matplotlib` & `seaborn` for data visualization
  - `sklearn` for machine learning algorithms
- **Random Forest Classifier** for model training

## Dataset
The dataset used in this project contains health-related information about patients, such as:
- Pregnancies
- Glucose levels
- Blood Pressure (BP)
- Skin Thickness
- Insulin levels
- BMI
- Diabetes Pedigree Function (DPF)
- Age
- Outcome (1 = Diabetic, 0 = Non-Diabetic)

You can download the dataset from the [Kaggle Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database) or use the provided `diabetes.csv` file in this repository.

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset:**
   Place the `diabetes.csv` dataset in the project directory.

## Usage

1. **Run the Code:**
   You can execute the script by running:
   ```bash
   python diabetes_prediction.py
   ```

2. **Real-Time Predictions:**
   After training, the script will prompt you to enter health information (e.g., glucose level, BMI, etc.) for real-time diabetes prediction.

3. **Visualization:**
   The script generates multiple visualizations:
   - A **Heatmap** to show correlations between the dataset features.
   - **Pie Chart** and **Bar Graph** to show the distribution of diabetic and non-diabetic individuals in the dataset.

## Model Evaluation

The model achieved an accuracy of **X%** on the test data. The performance of the Random Forest Classifier was evaluated using standard metrics like accuracy, precision, and recall.

Key steps in the model pipeline:
- **Data Preprocessing:** Handling missing/zero values and feature renaming.
- **Train-Test Split:** 70% of the data used for training and 30% for testing.
- **Model Training:** Random Forest Classifier was used with default parameters.
- **Accuracy Evaluation:** The model accuracy was computed using the test set.

## Future Improvements

- **Hyperparameter Tuning:** Improve the model’s performance by fine-tuning parameters.
- **Testing with Other Models:** Compare performance with other algorithms such as SVM, Logistic Regression, and Neural Networks.
- **Larger Datasets:** Expanding the dataset to include more features and records could enhance the model's accuracy and generalizability.

## Contributing

If you want to contribute to this project, feel free to submit issues or pull requests. Any suggestions or improvements are welcome.

1. Fork the repo.
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Open a pull request.
