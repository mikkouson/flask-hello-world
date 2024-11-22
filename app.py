from flask import Flask, jsonify
from supabase import create_client, Client
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import websocket
import json
import os
from dotenv import load_dotenv
import threading
from datetime import datetime, timedelta
from flask_cors import CORS
load_dotenv()

app = Flask(__name__)
CORS(app)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def prepare_training_data():
    """Fetch and prepare training data from Supabase"""
    try:
        # Get historical usage data for the last 90 days
        ninety_days_ago = (datetime.now() - timedelta(days=90)).isoformat()
        
        # Fetch items_used data
        usage_data = supabase.table("items_used")\
            .select("item_id,quantity,created_at")\
            .gte("created_at", ninety_days_ago)\
            .execute()
        
        if not usage_data.data:
            raise ValueError("No usage data found")
            
        # Convert to DataFrame
        usage_df = pd.DataFrame(usage_data.data)
        
        # Group by item_id and aggregate
        usage_stats = usage_df.groupby('item_id').agg({
            'quantity': ['sum', 'mean', 'std', 'count'],
            'created_at': lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).days
        }).reset_index()
        
        # Flatten column names
        usage_stats.columns = ['item_id', 'total_used', 'avg_usage', 'std_usage', 'usage_frequency', 'days_span']
        
        # Calculate daily usage rate
        usage_stats['daily_usage_rate'] = usage_stats['total_used'] / usage_stats['days_span'].clip(lower=1)
        
        # Fetch current inventory data
        inventory_data = supabase.table("inventory")\
            .select("id,quantity,name,description")\
            .execute()
            
        if not inventory_data.data:
            raise ValueError("No inventory data found")
            
        inventory_df = pd.DataFrame(inventory_data.data)
        
        # Merge inventory with usage statistics
        final_df = pd.merge(
            inventory_df,
            usage_stats,
            left_on='id',
            right_on='item_id',
            how='left'
        )
        
        # Fill NaN values for items with no usage history
        final_df = final_df.fillna({
            'total_used': 0,
            'avg_usage': 0,
            'std_usage': 0,
            'usage_frequency': 0,
            'daily_usage_rate': 0,
            'days_span': 0
        })
        
        return final_df
    
    except Exception as e:
        print(f"Error preparing training data: {e}")
        raise

def train_model(df):
    """Train Random Forest model and return model with evaluation metrics"""
    try:
        # Prepare features
        features = ['quantity', 'total_used', 'avg_usage', 'std_usage', 
                   'usage_frequency', 'daily_usage_rate']
        X = df[features]
        y = df['daily_usage_rate']  # Predict daily usage rate
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        return model, metrics, features
    
    except Exception as e:
        print(f"Error training model: {e}")
        raise

def predict_inventory_needs(model, features, df):
    """Generate inventory predictions and recommendations"""
    try:
        # Predict daily usage rate
        X_pred = df[features]
        predicted_daily_usage = model.predict(X_pred)
        
        # Calculate recommendations for 30-day supply
        results = []
        for idx, row in df.iterrows():
            predicted_monthly_usage = predicted_daily_usage[idx] * 30
            current_quantity = row['quantity']
            
            # Calculate restock amount (include 20% safety buffer)
            safety_buffer = 1.2
            recommended_stock = predicted_monthly_usage * safety_buffer
            restock_amount = max(0, recommended_stock - current_quantity)
            
            results.append({
                'id': row['id'],
                'name': row['name'],
                'current_quantity': current_quantity,
                'predicted_daily_usage': predicted_daily_usage[idx],
                'predicted_monthly_usage': predicted_monthly_usage,
                'recommended_restock': round(restock_amount)
            })
            
        return results
    
    except Exception as e:
        print(f"Error generating predictions: {e}")
        raise

@app.route('/api/predict', methods=['GET'])
def predict_inventory():
    try:
        # Prepare data
        df = prepare_training_data()
        
        # Train model and get metrics
        model, metrics, features = train_model(df)
        
        # Generate predictions
        predictions = predict_inventory_needs(model, features, df)
        
        # Prepare response
        response = {
            'predictions': predictions,
            'model_metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket handling code remains the same
def on_message(ws, message):
    print("Received message:", message)
    try:
        data = json.loads(message)
        print(f"Event type: {data.get('event')}")
        print(f"Payload: {data.get('payload')}")
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### WebSocket Connection Closed ###")

def on_open(ws):
    print("WebSocket Connection Opened")
    subscribe_msg = {
        "event": "phx_join",
        "topic": "realtime:public:inventory",
        "payload": {},
        "ref": "1"
    }
    ws.send(json.dumps(subscribe_msg))

def start_websocket():
    ws_url = SUPABASE_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/realtime/v1/websocket'
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        ws_url,
        header={"apikey": SUPABASE_KEY},
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

if __name__ == '__main__':
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()
    app.run(debug=True)