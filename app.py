import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def prepare_data():
    """Fetch and prepare inventory and usage data"""
    try:
        # Fetch inventory data
        inventory_data = supabase.table("inventory").select("*").execute()
        inventory_df = pd.DataFrame(inventory_data.data)

        # Fetch usage data
        usage_data = supabase.table("items_used").select("*").execute()
        usage_df = pd.DataFrame(usage_data.data)

        # Group usage data by item_id and date
        usage_pivot = usage_df.pivot_table(
            index='item_id', 
            columns='created_at', 
            values='quantity', 
            aggfunc='sum'
        ).reset_index()

        # Fill missing dates with 0
        usage_pivot = usage_pivot.fillna(0)

        # Merge inventory with usage data
        merged_df = pd.merge(
            inventory_df, 
            usage_pivot, 
            left_on='id', 
            right_on='item_id', 
            how='left'
        )

        return merged_df

    except Exception as e:
        print(f"Error preparing data: {e}")
        raise

def train_predict_model(merged_df):
    """Train predictive model and generate recommendations"""
    try:
        # Prepare features and target
        date_columns = [col for col in merged_df.columns if isinstance(col, str) and col.startswith('20')]
        
        # Create features matrix
        X = merged_df[date_columns]
        y = X.sum(axis=1)  # Total usage as target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()

        # Predict next month's usage
        predicted_usage = model.predict(X)
        
        # Generate recommendations
        recommendations = []
        for idx, row in merged_df.iterrows():
            predicted_monthly_usage = predicted_usage[idx]
            current_quantity = row['quantity']
            
            # Recommend restock with 20% safety buffer
            recommended_restock = max(0, int(predicted_monthly_usage * 1.2 - current_quantity))
            
            recommendations.append({
                'item_id': row['id'],
                'item_name': row['name'],
                'current_quantity': current_quantity,
                'predicted_monthly_usage': int(predicted_monthly_usage),
                'recommended_restock': recommended_restock
            })

        return recommendations, metrics

    except Exception as e:
        print(f"Error in model training: {e}")
        raise

def main():
    # Prepare data
    data = prepare_data()
    
    # Train model and get recommendations
    recommendations, metrics = train_predict_model(data)
    
    # Print results
    print("\n--- Inventory Predictions and Recommendations ---")
    for rec in recommendations:
        print(f"Item: {rec['item_name']} (ID: {rec['item_id']})")
        print(f"Current Quantity: {rec['current_quantity']}")
        print(f"Predicted Monthly Usage: {rec['predicted_monthly_usage']}")
        print(f"Recommended Restock: {rec['recommended_restock']}")
        print("---")
    
    print("\n--- Model Performance Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()