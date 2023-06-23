import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from skopt import BayesSearchCV
from ydata_profiling import ProfileReport

# Przykładowe dane
filepath = 'ASI_final/asi-kedro/data/01_raw/ds_salaries.csv'
data = pd.read_csv(filepath)

# print(data['job_title'].unique())



# Wykonaj analizę danych
profile = ProfileReport(data)
# Możesz również użyć innych narzędzi do analizy danych


ohe = OneHotEncoder()
feature_array = ohe.fit_transform(data[['job_title','experience_level', 'employment_type', 'employee_residence', 'company_location','salary_currency','company_size']]).toarray()

feature_labels = np.concatenate(ohe.categories_)

features = pd.DataFrame(feature_array, columns=feature_labels)

data_encoded = pd.concat([data, features], axis=1)

final = data_encoded.drop(['job_title','experience_level', 'employment_type', 'employee_residence', 'company_location','salary_currency','company_size'], axis=1)

# Podziel dane na zbiór cech (X) i zmienną docelową (y)
X = final.drop('salary', axis=1)
y = final['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Użyj grid search do znalezienia optymalnych hyperparametrów dla Random Forest Classifier
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier()

grid_search_rf = GridSearchCV(estimator = rf_model, param_grid = param_grid_rf, cv = 3, verbose= 2, n_jobs = 4)
grid_search_rf.fit(X_train, y_train)

best_rf_model = grid_search_rf.best_estimator_

print(grid_search_rf.best_params_)
print(best_rf_model)
print (f'Train Accuracy - : {grid_search_rf.score(X_train, y_train):.3f}')
print (f'Test Accuracy - : {grid_search_rf.score(X_test, y_test):.3f}')


