# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


"""# Import Data"""

excel_file = 'Winequality_dataset_final.xlsx'
sheet_name = 'data'

data  = pd.read_excel(excel_file,
                    sheet_name = sheet_name,
                    usecols = 'A:K',
                    header = 0)

col_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'sulphates', 'alcohol', 'quality']

"""# Min max values"""

fixed_min = data['fixed acidity'].min()
fixed_max = data['fixed acidity'].max()
volatile_min = data['volatile acidity'].min()
volatile_max = data['volatile acidity'].max()
citric_min = data['citric acid'].min()
citric_max = data['citric acid'].max()
residual_min = data['residual sugar'].min()
residual_max = data['residual sugar'].max()
chlorides_min = data['chlorides'].min()
chlorides_max = data['chlorides'].max()
free_min = data['free sulfur dioxide'].min()
free_max = data['free sulfur dioxide'].max()
total_min = data['total sulfur dioxide'].min()
total_max = data['total sulfur dioxide'].max()
density_min = data['density'].min()
density_max = data['density'].max()
sulphates_min = data['sulphates'].min()
sulphates_max = data['sulphates'].max()
alcohol_min = data['alcohol'].min()
alcohol_max = data['alcohol'].max()
quality_min = data['quality'].min()
quality_max = data['quality'].max()


"""# Normalization"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Applying scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
            'total sulfur dioxide', 'density', 'sulphates', 'alcohol', 'quality']
data[num_vars] = scaler.fit_transform(data[num_vars])

"""# Data Split"""

#split dataset in features and target variable
feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','sulphates','alcohol']
X = data[feature_cols] 
y = data.quality
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

"""# Decision Tree Classifier Tree"""

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

"""# Prediction Using Decision Tree"""

#Predict the response for test dataset
y_pred = clf.predict(X_test)

"""# Performance of the model"""

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# Decision Tree Optimization
"""

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


"""#Thank You"""