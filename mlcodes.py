#DECISION TREE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the CSV file into a DataFrame
data = pd.read_csv("../salaries.csv")  
# Replace "jobs.csv" with your file's name
# Select the columns for features (X) and the target (y)
X = data[["company", "job", "degree"]]
y = data["salary_more_then_100k"]
# Convert categorical variables (company, job, degree) into numerical format
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier()
# Train the classifier on the training data
decision_tree_classifier.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = decision_tree_classifier.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))

********************************************************************************************************************

LINEAR REGRESSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("Experience-Salary.csv",converters={"exp(in months)":int()})
# y = Bo + Bi X
x= data['exp(in months)']
print(x)
y = data['salary(in thousands)']
plt.scatter(x,y)
plt.show
mean_x = np.mean(x)
mean_y = np.mean(y)
print(mean_y)
numerator =0
denominatr =0
L = len(x)
for i in range(L):
    ab = (x[i]-mean_x)*(y[i]-mean_y)
    cd = (x[i]-mean_x)**2
    numerator += ab
    denominatr += cd 
ans = numerator/denominatr
print(ans)
reg = mean_y - (ans * mean_x)
max_X = np.max(x) +100
min_y = np.min(y) -100
X = np.linspace(max_X,min_y,100)
Y = reg + ans*X
plt.plot(X,Y,color='green')
plt.scatter(x,y)
plt.xlabel("op")
plt.ylabel('sal')
plt.legend()
plt.show()

********************************************************************************************************************

#LOGISTIC REGRESSION
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the breast cancer dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, names=column_names)
# Split the dataset into features and target
X = data.drop(["ID", "Diagnosis"], axis=1)
y = data["Diagnosis"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression()
# Train the classifier on the training data
logistic_classifier.fit(X_train, y_train)
# Make predictions on the test data
y_pred = logistic_classifier.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
# Display classification report and confusion matrix
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


********************************************************************************************************************

#SUPPORT VECTOR MACHINES
#SVM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the breast cancer dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ["ID", "Diagnosis"] + [f"Feature_{i}" for i in range(1, 31)]
data = pd.read_csv(url, names=column_names)
# Split the dataset into features and target
X = data.drop(["ID", "Diagnosis"], axis=1)
y = data["Diagnosis"]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an SVM classifier
svm_classifier = SVC(kernel='linear')
# Train the classifier on the training data
svm_classifier.fit(X_train, y_train)
# Make predictions on the training data
y_train_pred = svm_classifier.predict(X_train)
# Calculate the accuracy of the model on the training data
train_accuracy = accuracy_score(y_train, y_train_pred)
# Display classification report and confusion matrix for training data
print("Training Data:")
print("Accuracy:", train_accuracy)
print("Classification Report:")
print(classification_report(y_train, y_train_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
# Make predictions on the test data
y_test_pred = svm_classifier.predict(X_test)
# Calculate the accuracy of the model on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
# Display classification report and confusion matrix for test data
print("\nTesting Data:")
print("Accuracy:", test_accuracy)
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))



********************************************************************************************************************
#ENSEMBLE LEARNING
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset
data = pd.read_csv("boston.csv")

# Split the dataset into features and target
X = data.drop("medv", axis=1)
y = data["medv"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the regressor on the training data
random_forest_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = random_forest_regressor.predict(X_test)

# Calculate the Mean Squared Error (MSE) and R-squared (R2) score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display detailed outputs
print("Random Forest Regressor Results:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# Feature Importance
feature_importance = random_forest_regressor.feature_importances_
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# Individual Tree Predictions
individual_tree_predictions = [estimator.predict(X_test) for estimator in random_forest_regressor.estimators_]

print("\nPredictions from Individual Trees (First 5 trees):")
for i, prediction in enumerate(individual_tree_predictions[:5]):
    print(f"Tree {i + 1}: {prediction}")



********************************************************************************************************************
#PCA
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
# Load the Breast Cancer dataset
data = load_breast_cancer().data

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)
# Create a PCA instance with 2 components
n_components = 2
pca = PCA(n_components=n_components)
# Fit PCA to the scaled data
pca.fit(scaled_data)
# Transform the data to its principal components
transformed_data = pca.transform(scaled_data)
# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Principal components
principal_components = pca.components_
print("Principal Components:", principal_components)

********************************************************************************************************************
#GRADIENT
# example of plotting a gradient descent search on a one-dimensional function
from numpy import asarray
from numpy import arange
from numpy.random import rand
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# derivative of objective function
def derivative(x):
	  # Calculate the derivative of the provided objective function
      h = 1e-6  # A small number for numerical differentiation
      return (objective(x + h) - objective(x - h)) / (2*h)

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size):
	# track all solutions
	solutions, scores = list(), list()
	# generate an initial point
	solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	# run the gradient descent
	for i in range(n_iter):
		# calculate gradient
		gradient = derivative(solution)
		# take a step
		solution = solution - step_size * gradient
		# evaluate candidate point
		solution_eval = objective(solution)
		# store solution
		solutions.append(solution)
		scores.append(solution_eval)
		# report progress
		print('>%d f(%s) = %.5f' % (i, solution, solution_eval))
	return [solutions, scores]

# define range for input
bounds = asarray([[-1.0, 1.0]])
# define the total iterations
n_iter = 100
# define the step size
step_size = 0.01
# perform the gradient descent search
solutions, scores = gradient_descent(objective, derivative, bounds, n_iter, step_size)
# sample input range uniformly at 0.1 increments
inputs = arange(bounds[0,0], bounds[0,1]+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# plot the solutions found
pyplot.plot(solutions, scores, '.-', color='red')
# show the plot
pyplot.show()

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
	
	# Calculating the loss or cost
	cost = np.sum((y_true-y_predicted)**2) / len(y_true)
	return cost

# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001, 
					stopping_threshold = 1e-6):
	
	# Initializing weight, bias, learning rate and iterations
	current_weight = 0.1
	current_bias = 0.01
	iterations = iterations
	learning_rate = learning_rate
	n = float(len(x))
	
	costs = []
	weights = []
	previous_cost = None
	
	# Estimation of optimal parameters 
	for i in range(iterations):
		
		# Making predictions
		y_predicted = (current_weight * x) + current_bias
		
		# Calculating the current cost
		current_cost = mean_squared_error(y, y_predicted)

		# If the change in cost is less than or equal to 
		# stopping_threshold we stop the gradient descent
		if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
			break
		
		previous_cost = current_cost

		costs.append(current_cost)
		weights.append(current_weight)
		
		# Calculating the gradients
		weight_derivative = -(2/n) * sum(x * (y-y_predicted))
		bias_derivative = -(2/n) * sum(y-y_predicted)
		
		# Updating weights and bias
		current_weight = current_weight - (learning_rate * weight_derivative)
		current_bias = current_bias - (learning_rate * bias_derivative)
				
		# Printing the parameters for each 1000th iteration
		print(f"Iteration {i+1}: Cost {current_cost}, Weight \
		{current_weight}, Bias {current_bias}")
	
	
	# Visualizing the weights and cost at for all iterations
	plt.figure(figsize = (8,6))
	plt.plot(weights, costs)
	plt.scatter(weights, costs, marker='o', color='red')
	plt.title("Cost vs Weights")
	plt.ylabel("Cost")
	plt.xlabel("Weight")
	plt.show()
	
	return current_weight, current_bias


def main():
	
	# Data
	X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
		55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
		45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
		48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
	Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
		78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
		55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
		60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

	# Estimating weight and bias using gradient descent
	estimated_weight, estimated_bias = gradient_descent(X, Y, iterations=2000)
	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

	# Making predictions using estimated parameters
	Y_pred = estimated_weight*X + estimated_bias

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(X, Y, marker='o', color='red')
	plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

	
if __name__=="__main__":
	main()


********************************************************************************************************************

#DATA PREPROCESSION

# importing libraries 
import numpy as np 
import pandas as pd
import matplotlib as plt 
# dataset read 
data = pd.read_csv('data.csv')
df = pd.DataFrame(data)
df
#1. Ireplace missing values
# Perform mean imputation for the 'Salary' column
mean_salary = df['Salary'].mean()
mean_age = df['Age'].mean()
df['Salary'].fillna(mean_salary, inplace=True)
df['Age'].fillna(mean_age, inplace=True)
df
# set decimal precision to 2
df['Salary'] = df['Salary'].round(2)
df['Age'] = (df['Age']).astype(int)
df

# 2. Anomaly Detection - 
new_data = {
    'Country': 'pakistan',
    'Age': 140,
    'Salary': 1000000,
    'Purchased': 'No',
}
new_row = pd.DataFrame([new_data])
# Append the new row to the original DataFrame
df = pd.concat([df, new_row], ignore_index=True)
df1 =df.copy()
# # Calculate the z-scores for the 'Age' and 'Salary' columns
df1['Age_ZScore'] = (df['Age'] - df['Age'].mean()) / df['Age'].std(ddof=0)
df1['Salary_ZScore'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std(ddof=0)
df
# Find the tuples with the maximum absolute z-score values
max_abs_age_zscore = df1['Age_ZScore'].abs().max()
max_abs_salary_zscore = df1['Salary_ZScore'].abs().max()
# Filter the DataFrame to include only the tuples with maximum absolute z-score values
max_abs_age_tuples = df1[df1['Age_ZScore'].abs() == max_abs_age_zscore]
max_abs_salary_tuples = df1[df1['Salary_ZScore'].abs() == max_abs_salary_zscore]
max_abs_age_tuples
max_abs_salary_tuples
hence Anomaly detected for this given tuple with salary and age deviates from normal data according to z_scores
3. Standardization 
#standardization (z-score) formula
#z = (x - μ) / σ

# # Calculate the z-scores for the 'Age' and 'Salary' columns
df2 = df.copy()
df2['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std(ddof=0)
df2['Salary'] = (df['Salary'] - df['Salary'].mean()) / df['Salary'].std(ddof=0)
df2
4. Normalization
# Normalize 'Age' and 'Salary' columns using Min-Max scaling
df3 = df.copy()
df3['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df3['Salary'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())
df3
Encoding
df_encoded = pd.get_dummies(df, columns=['Country', 'Purchased'], drop_first=True)
df_encoded


