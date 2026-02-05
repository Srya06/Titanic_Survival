import joblib
import pandas as pd
import numpy as np

class TitanicPredictor:
    def __init__(self, model_path='saved_models/logistic_model.pkl',
                 scaler_path='saved_models/scaler.pkl',
                 features_path='saved_models/feature_columns.pkl'):
        
        try:
            # Load trained model and preprocessing objects
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)
            
            print(f"Model loaded successfully with {len(self.feature_columns)} features")
            print(f"Features: {self.feature_columns}")
            
        except FileNotFoundError as e:
            print(f"Warning: {e}. Using default configuration.")
            self.model = None
            self.scaler = None
            self.feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
    
    def preprocess_input(self, input_dict):
        """Preprocess single input using YOUR logic"""
        
        # Create DataFrame from input
        df = pd.DataFrame([input_dict])
        
        print(f"Raw input: {input_dict}")
        
        # === APPLY YOUR CLEANING LOGIC ===
        # Drop unnecessary columns if they exist
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
        for col in cols_to_drop:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        
        # Convert Sex to binary (1=male, 0=female) - YOUR LOGIC
        if 'Sex' in df.columns:
            df["Sex"] = np.where(df["Sex"].str.lower() == "male", 1, 0)
        
        # Fill missing Age with mean if not provided
        if 'Age' in df.columns:
            if df['Age'].isnull().any():
                # Use average age (29.7) if not provided
                df['Age'] = df['Age'].fillna(29.7)
        
        # Map Embarked if provided
        if 'Embarked' in df.columns:
            embarked_map = {'S': 0, 'C': 1, 'Q': 2}
            df['Embarked'] = df['Embarked'].map(embarked_map).fillna(0)
        
        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                # Add missing columns with default values
                if col == 'Sex':
                    df[col] = 0  # Default female
                elif col == 'Age':
                    df[col] = 29.7  # Average age
                else:
                    df[col] = 0
        
        # Reorder columns to match training
        df = df[self.feature_columns]
        
        print(f"Processed features: {df.values}")
        
        # Scale using saved scaler
        if self.scaler:
            df_scaled = self.scaler.transform(df)
        else:
            df_scaled = df.values
        
        return df_scaled
    
    def predict(self, input_dict):
        """Make prediction for single input"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_dict)
            
            if self.model:
                # Make prediction
                prediction = self.model.predict(processed_data)[0]
                probability = self.model.predict_proba(processed_data)[0][prediction]
                
                # Get feature importance for this prediction
                feature_importance = self.get_feature_impact(processed_data[0])
            else:
                # Demo prediction (fallback)
                prediction = 1 if input_dict.get('Sex') == 0 else 0
                probability = 0.75
                feature_importance = {}
            
            # Return results
            result = {
                'survived': bool(prediction),
                'survival_status': "Survived 🎉" if prediction == 1 else "Did Not Survive 💀",
                'probability': round(probability * 100, 2),
                'prediction': prediction,
                'feature_impact': feature_importance
            }
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'error': str(e),
                'survived': False,
                'survival_status': "Error in prediction",
                'probability': 0.0
            }
    
    def get_feature_impact(self, features):
        """Calculate how each feature impacted the prediction"""
        if self.model is None:
            return {}
        
        # Get model coefficients
        coefficients = self.model.coef_[0]
        
        # Calculate impact
        impact = {}
        for i, feature in enumerate(self.feature_columns):
            # Impact = feature value * coefficient
            feature_value = features[i]
            coefficient = coefficients[i]
            impact_score = feature_value * coefficient
            
            # Determine direction
            if coefficient > 0:
                direction = "increased" if feature_value > 0 else "decreased"
            else:
                direction = "decreased" if feature_value > 0 else "increased"
            
            impact[feature] = {
                'value': round(feature_value, 2),
                'coefficient': round(coefficient, 4),
                'impact': round(impact_score, 4),
                'direction': direction,
                'effect': "Helped survival" if impact_score > 0 else "Reduced survival chances"
            }
        
        # Sort by absolute impact
        return dict(sorted(impact.items(), key=lambda x: abs(x[1]['impact']), reverse=True))