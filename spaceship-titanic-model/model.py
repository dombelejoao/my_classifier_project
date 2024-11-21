import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Enhanced feature engineering
def preprocess_data(df):
    # Extract cabin information
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
   
    # More sophisticated spending features
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    df['SpendingVariance'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].std(axis=1)
    df['SpendingRatio'] = df['TotalSpending'] / (df['Age'] + 1)  # Prevent division by zero
   
    # Group size feature
    df['GroupSize'] = df['PassengerId'].str.split('_').str[0].map(df['PassengerId'].str.split('_').str[0].value_counts())
   
    return df

train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# Separate features and target
X = train_data.drop(['Transported', 'PassengerId', 'Name', 'Cabin'], axis=1)
y = train_data['Transported']

# Column types
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                   'TotalSpending', 'SpendingVariance', 'SpendingRatio', 'GroupSize']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_columns),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_columns)
    ])

# Classifier with GridSearch
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# Validate on holdout set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Prepare submission
test_ids = test_data['PassengerId']
X_test = test_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1)
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': predictions
})
submission.to_csv('submission.csv', index=False)
