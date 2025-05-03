import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")  # Ensure correct delimiter

# Define encoding dictionary
encoding_map = {
    "school": {"GP": 1, "MS": 0},
    "sex": {"M": 1, "F": 0},
    "address": {"U": 1, "R": 0},
    "famsize": {"LE3": 1, "GT3": 0},
    "Pstatus": {"T": 1, "A": 0},
    "Mjob": {"teacher": 4, "health": 3, "services": 2, "at_home": 1, "other": 0},
    "Fjob": {"teacher": 4, "health": 3, "services": 2, "at_home": 1, "other": 0},
    "reason": {"home": 3, "reputation": 2, "course": 1, "other": 0},
    "guardian": {"mother": 2, "father": 1, "other": 0},
    "schoolsup": {"yes": 1, "no": 0},
    "famsup": {"yes": 1, "no": 0},
    "paid": {"yes": 1, "no": 0},
    "activities": {"yes": 1, "no": 0},
    "nursery": {"yes": 1, "no": 0},
    "higher": {"yes": 1, "no": 0},
    "internet": {"yes": 1, "no": 0},
    "romantic": {"yes": 1, "no": 0}
}

# Apply encoding
df.replace(encoding_map, inplace=True)
df = df.infer_objects(copy=False)  # Suppress FutureWarning

# Convert categorical data to numerical values
df_numeric = df.apply(lambda col: pd.factorize(col)[0] if col.dtype == 'object' else col)

# Convert to NumPy array and save as text file
np.savetxt("student-mat-no.txt", df_numeric.to_numpy(), fmt="%.4f")

# Identify numerical columns
numerical_features = df_numeric.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Compute correlation with the target variable (G3)
correlation = df_numeric[numerical_features].corr()["G3"].abs().sort_values(ascending=False)

# Select key numerical features (strongly correlated with G3) with correlation above 0.3
key_numerical_features = [feature for feature in correlation.index if correlation[feature] > 0.3]

# Descriptive statistics for key numerical features
print("Descriptive Statistics for Key Numerical Features:\n")
print(df_numeric[key_numerical_features].describe())

# Visualization: Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df_numeric[key_numerical_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Key Numerical Features")
plt.show()

# Histograms for key numerical features
df_numeric[key_numerical_features].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Histograms of Key Numerical Features")
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_numeric[key_numerical_features], orient="v", palette="Set2")
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.title("Box Plots of Key Numerical Features")
plt.ylim(df_numeric[key_numerical_features].min().min() - 1, df_numeric[key_numerical_features].max().max() + 1)  # Adjust scale
plt.show()


def process(feature, degree):
    """Generate polynomial features up to the given degree."""
    phi_matrix = [np.ones(feature.shape[0])]  # Bias term (column of ones)
    for i in range(feature.shape[1]):  # Loop over each feature
        for d in range(1, degree + 1):  # Loop over polynomial degrees
            phi_matrix.append(np.power(feature[:, i], d))  # Compute power
    return np.column_stack(phi_matrix)  # Convert list to 2D NumPy array


def load_and_split_data(file_path, split_ratio=0.8):
    """Load and randomly split data into training and testing sets."""
    data = np.loadtxt(file_path)
    train, test = train_test_split(data, test_size=(1 - split_ratio), random_state=None)  # Different each run
    return train, test


def linear_regression(file_path, degree, _lambda):
    """Perform polynomial regression using the closed-form solution."""
    # Load and split data
    training, testing = load_and_split_data(file_path)

    # Extract features and target variable
    phi_matrix = training[:, :-1]  # All columns except last (features)
    t_vector = training[:, -1]  # Last column (target values)

    # Process features with polynomial expansion
    phi_matrix = process(phi_matrix, degree)

    # Identity matrix for regularization
    identity_matrix = np.eye(phi_matrix.shape[1])

    # Compute weights using the closed-form solution
    w_vector = np.linalg.pinv(phi_matrix.T @ phi_matrix + _lambda * identity_matrix) @ phi_matrix.T @ t_vector

    # Print learned weights
    for i, w in enumerate(w_vector):
        print(f"w{i} = {w:.4f}")

    # Prepare testing data
    test_features = testing[:, :-1]  # All columns except last (features)
    target_values = testing[:, -1]  # Last column (actual values)
    test_features = process(test_features, degree)

    # Predict output
    output = test_features @ w_vector

    # Compute squared error for each prediction
    squared_errors = []
    for i, (predicted, actual) in enumerate(zip(output, target_values), start=1):
        squared_error = (actual - predicted) ** 2
        squared_errors.append(squared_error)
        print(f"ID={i}, output={predicted:.4f}, target value={actual:.4f}, squared error={squared_error:.4f}")

    # Compute Mean Squared Error (MSE)
    mse = np.mean(squared_errors)
    print(f"\nMean Squared Error (MSE): {mse:.4f}")

    # Compute R-squared (R²) Score
    ss_total = np.sum((target_values - np.mean(target_values)) ** 2)  # Total Sum of Squares (SST)
    ss_residual = np.sum((target_values - output) ** 2)  # Residual Sum of Squares (SSE)
    r_squared = 1 - (ss_residual / ss_total)
    
    print(f"R-squared Score (R²): {r_squared:.4f}")

    # Actual vs Predicted Grades Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(target_values, output, color='blue', alpha=0.5)  # Scatter plot of Actual vs Predicted
    plt.plot([min(target_values), max(target_values)], [min(target_values), max(target_values)], color='red', linestyle='--')  # Diagonal line (perfect prediction line)
    plt.title('Actual vs Predicted Grades')
    plt.xlabel('Actual Grades')
    plt.ylabel('Predicted Grades')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    file_path = "student-mat-no.txt"  # Path to data file
    linear_regression(file_path, degree=2, _lambda=0)
