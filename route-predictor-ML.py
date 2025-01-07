import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

class RouteOptimizationModel:
    def __init__(self):
        self.emissions_model = None
        self.time_model = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        
        features = ['distance', 'traffic_density', 'temperature', 'precipitation', 
                   'vehicle_weight', 'fuel_efficiency']
        targets = ['emissions', 'travel_time']
        
        X = df[features]
        y_emissions = df['emissions']
        y_time = df['travel_time']
        
        # Split data
        X_train, X_test, y_emissions_train, y_emissions_test, y_time_train, y_time_test = \
            train_test_split(X, y_emissions, y_time, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_test_scaled, y_emissions_train, 
                y_emissions_test, y_time_train, y_time_test)
    
    def train(self, X_train, y_emissions_train, y_time_train):
        # Train emissions model
        self.emissions_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.emissions_model.fit(X_train, y_emissions_train)
        
        # Train travel time model
        self.time_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.time_model.fit(X_train, y_time_train)
    
    def evaluate(self, X_test, y_emissions_test, y_time_test):
        emissions_pred = self.emissions_model.predict(X_test)
        time_pred = self.time_model.predict(X_test)
        
        emissions_mae = mean_absolute_error(y_emissions_test, emissions_pred)
        time_mse = mean_squared_error(y_time_test, time_pred)
        
        return {
            'emissions_mae': emissions_mae,
            'time_mse': time_mse,
            'emissions_predictions': emissions_pred,
            'time_predictions': time_pred
        }
    
    def predict_route(self, route_features):
        # Scale input features
        route_features_scaled = self.scaler.transform(route_features)
        
        emissions = self.emissions_model.predict(route_features_scaled)
        time = self.time_model.predict(route_features_scaled)
        
        return {
            'predicted_emissions': emissions[0],
            'predicted_time': time[0]
        }
    
    def save_model(self, path):
        model_data = {
            'emissions_model': self.emissions_model,
            'time_model': self.time_model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        model_data = joblib.load(path)
        self.emissions_model = model_data['emissions_model']
        self.time_model = model_data['time_model']
        self.scaler = model_data['scaler']

def train_and_save_model():
    model = RouteOptimizationModel()
    
    (X_train, X_test, y_emissions_train, y_emissions_test,
     y_time_train, y_time_test) = model.prepare_data('route_data.csv')
    
    model.train(X_train, y_emissions_train, y_time_train)
    
    metrics = model.evaluate(X_test, y_emissions_test, y_time_test)
    print(f"Emissions MAE: {metrics['emissions_mae']:.2f}")
    print(f"Time MSE: {metrics['time_mse']:.2f}")
    
    model.save_model('route_optimization_model.joblib')
    
    return model

def predict_new_route(model, route_data): 
    predictions = model.predict_route(route_data)
    return predictions
