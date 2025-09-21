from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class Data_trainer:
    @staticmethod
    def train_random_forest(X, y, test_size=0.2, random_state=42, **kwargs):
        """Обучает модель случайного леса"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = RandomForestClassifier(**kwargs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    @staticmethod
    def train_linear_regression(X, y, test_size=0.2, random_state=42, **kwargs):
        """Обучает модель линейной регрессии"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = LinearRegression(**kwargs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return model, mse
    
    @staticmethod
    def train_logistic_regression(X, y, test_size=0.2, random_state=42, **kwargs):
        """Обучает модель логистической регрессии"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
