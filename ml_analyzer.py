import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
import joblib
from google.cloud import aiplatform
from google.cloud import storage
import os
from google.oauth2 import service_account

class SmartHomeML:
    def __init__(self, mongodb_uri):
        self.mongodb_uri = mongodb_uri
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        
    def setup_google_auth(self, credentials_path):
        """Setup Google Cloud authentication"""
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/aiplatform"
                ]
            )
            
            aiplatform.init(
                credentials=credentials,
                project='unified-post-446702-c7',  # Update with your project ID
                location='asia-southeast1'
            )
            return True
        except Exception as e:
            print(f"Authentication setup failed: {str(e)}")
            return False

    def save_model_local(self, model, model_name):
        """Save model locally before uploading to GCS"""
        local_path = f"/tmp/{model_name}"
        tf.saved_model.save(model, local_path)
        return local_path

    def upload_to_gcs(self, local_path, bucket_name, destination_blob_name):
        """Upload model to Google Cloud Storage"""
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    blob_path = os.path.join(destination_blob_name, relative_path)
                    
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_file_path)
                    
            return f"gs://{bucket_name}/{destination_blob_name}"
        except Exception as e:
            print(f"Failed to upload to GCS: {str(e)}")
            return None

    def deploy_to_vertex_ai(self, credentials_path, bucket_name):
        """Deploy models to Vertex AI"""
        if not self.setup_google_auth(credentials_path):
            return None

        endpoints = {}
        
        try:
            # Deploy light model
            if 'light' in self.models:
                light_local_path = self.save_model_local(
                    self.models['light'], 
                    'light_model'
                )
                
                light_gcs_path = self.upload_to_gcs(
                    light_local_path,
                    bucket_name,
                    'models/light'
                )
                
                if light_gcs_path:
                    light_model = aiplatform.Model.upload(
                        display_name="light_predictor",
                        artifact_uri=light_gcs_path,
                        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
                    )
                    endpoints['light'] = light_model.deploy(
                        machine_type="n1-standard-2",
                        min_replica_count=1,
                        max_replica_count=2
                    )

            # Deploy anomaly model
            if 'anomaly' in self.models:
                anomaly_local_path = self.save_model_local(
                    self.models['anomaly'], 
                    'anomaly_model'
                )
                
                anomaly_gcs_path = self.upload_to_gcs(
                    anomaly_local_path,
                    bucket_name,
                    'models/anomaly'
                )
                
                if anomaly_gcs_path:
                    anomaly_model = aiplatform.Model.upload(
                        display_name="anomaly_detector",
                        artifact_uri=anomaly_gcs_path,
                        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
                    )
                    endpoints['anomaly'] = anomaly_model.deploy(
                        machine_type="n1-standard-2",
                        min_replica_count=1,
                        max_replica_count=2
                    )

        except Exception as e:
            print(f"Deployment failed: {str(e)}")
            return None

        return endpoints

    def fetch_data(self):
        """Fetch and preprocess data from MongoDB"""
        client = MongoClient(self.mongodb_uri)
        db = client.smart_home
        collection = db.sensor_data
        
        data = list(collection.find({}))
        df = pd.DataFrame(data)
        
        # Flatten nested data
        data_df = pd.json_normalize(df['data'])
        df = pd.concat([df.drop('data', axis=1), data_df], axis=1)
        
        # Convert timestamp and add time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Handle missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Handle missing values in categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        return df
    
    def train_light_model(self, df):
        """Train light level prediction model (binary classification)"""
        light_df = df[df['device_type'] == 'light_sensor'].copy()
        
        # Feature engineering
        features = ['hour', 'day_of_week', 'month', 'is_weekend']
        X = light_df[features]
        y = light_df['is_dark'].astype(int)
        
        # Handle any remaining missing values in features
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=features)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define neural network model
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(len(features),)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with early stopping and learning rate reduction
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks
        )
        
        # Evaluate
        y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Light Model Accuracy: {accuracy}")
        
        self.models['light'] = model
        self.scalers['light'] = scaler
        self.imputers['light'] = imputer
        
        return model, scaler, accuracy, history
    
    def train_anomaly_detector(self, df):
        """Train anomaly detection model using Isolation Forest"""
        features_by_device = {
            'light_sensor': ['light_level', 'battery_level'],
            'motion_sensor': ['battery_level', 'motion_detected'],
            'door_sensor': ['battery_level', 'is_open']
        }
        
        anomaly_models = {}
        
        for device_type, features in features_by_device.items():
            device_df = df[df['device_type'] == device_type].copy()
            if device_df.empty:
                continue
                
            # Prepare features
            X = device_df[features]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Train Isolation Forest
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Store model, scaler, and imputer
            anomaly_models[device_type] = {
                'model': model,
                'scaler': scaler,
                'imputer': imputer,
                'features': features
            }
        
        self.models['anomaly'] = anomaly_models
        return anomaly_models
    
    def predict_light(self, data):
        """Make predictions using the light model"""
        if 'light' not in self.models:
            raise ValueError("Light model not trained")
            
        # Handle missing values
        imputer = self.imputers['light']
        data_imputed = pd.DataFrame(imputer.transform(data), columns=data.columns)
        
        # Scale input data
        scaler = self.scalers['light']
        data_scaled = scaler.transform(data_imputed)
            
        # Make prediction
        model = self.models['light']
        prediction = (model.predict(data_scaled) > 0.5).astype(int)
        
        return prediction
    
    def detect_anomalies(self, data, device_type):
        """Detect anomalies using the trained anomaly detection model"""
        if 'anomaly' not in self.models:
            raise ValueError("Anomaly detection model not trained")
            
        if device_type not in self.models['anomaly']:
            raise ValueError(f"No anomaly model trained for device type: {device_type}")
            
        # Get model info
        model_info = self.models['anomaly'][device_type]
        model = model_info['model']
        scaler = model_info['scaler']
        imputer = model_info['imputer']
        features = model_info['features']
        
        # Ensure data contains required features
        if not all(feature in data.columns for feature in features):
            raise ValueError(f"Data missing required features: {features}")
        
        # Handle missing values
        data_imputed = pd.DataFrame(imputer.transform(data[features]), columns=features)
        
        # Scale input data
        data_scaled = scaler.transform(data_imputed)
        
        # Detect anomalies (-1 for anomalies, 1 for normal data)
        predictions = model.predict(data_scaled)
        
        # Convert to boolean (True for anomalies)
        anomalies = predictions == -1
        
        return anomalies

def main():
    # Initialize
    ml_system = SmartHomeML(mongodb_uri="mongodb://localhost:27017/")
    
    # Fetch and prepare data
    df = ml_system.fetch_data()
    
    # Train models
    light_model, light_scaler, light_accuracy, light_history = ml_system.train_light_model(df)
    anomaly_models = ml_system.train_anomaly_detector(df)
    
    # Save models locally
    joblib.dump(light_scaler, '/tmp/light_scaler.pkl')
    joblib.dump(anomaly_models, '/tmp/anomaly_models.pkl')
    
    # Deploy to Vertex AI
    credentials_path = "smart-home-key.json"  # Update this
    bucket_name = "cpc357-ysproject-store-sensor-data"  # Update this
    
    try:
        endpoints = ml_system.deploy_to_vertex_ai(credentials_path, bucket_name)
        if endpoints:
            print("Models deployed successfully!")
            print("Endpoints:", endpoints)
        else:
            print("Model deployment failed. Using local models instead.")
            
            # Save models locally as backup
            tf.saved_model.save(light_model, '/tmp/light_model')
            print("Models saved locally in /tmp/")
    except Exception as e:
        print(f"Deployment error: {str(e)}")
        print("Falling back to local model saving...")
        tf.saved_model.save(light_model, '/tmp/light_model')

if __name__ == "__main__":
    main()