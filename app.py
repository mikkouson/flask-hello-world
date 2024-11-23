import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from supabase import create_client, Client
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import websocket
import json
import queue

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global queue for realtime updates
realtime_update_queue = queue.Queue()

def process_realtime_updates():
    """
    Background thread to process realtime updates to inventory
    """
    while True:
        try:
            # Block and wait for updates
            update = realtime_update_queue.get()
            
            # Log the update
            print(f"Processing Realtime Update: {update}")
            
            # You can add custom logic here for different event types
            if update.get('event_type') == 'INSERT':
                # Handle new inventory item
                pass
            elif update.get('event_type') == 'UPDATE':
                # Handle inventory item update
                pass
            elif update.get('event_type') == 'DELETE':
                # Handle inventory item deletion
                pass
            
            # Mark task as done
            realtime_update_queue.task_done()
        
        except Exception as e:
            print(f"Error processing realtime update: {e}")

def on_realtime_message(ws, message):
    """
    Callback for processing websocket realtime messages
    """
    try:
        data = json.loads(message)
        
        # Check for realtime event
        if data.get('event') in ['INSERT', 'UPDATE', 'DELETE']:
            update = {
                'event_type': data.get('event'),
                'table': data.get('table', 'unknown'),
                'payload': data.get('payload')
            }
            
            # Put update in queue for processing
            realtime_update_queue.put(update)
    
    except Exception as e:
        print(f"Error in realtime message handler: {e}")

def start_realtime_listener():
    """
    Start websocket for realtime Supabase updates
    """
    try:
        # Construct WebSocket URL
        ws_url = SUPABASE_URL.replace('https://', 'wss://').replace('http://', 'ws://') + '/realtime/v1/websocket'
        
        ws = websocket.WebSocketApp(
            ws_url,
            header={"apikey": SUPABASE_KEY},
            on_message=on_realtime_message
        )
        
        # Listen for changes in inventory table
        subscribe_msg = {
            "event": "phx_join",
            "topic": "realtime:public:inventory",
            "payload": {},
            "ref": "1"
        }
        
        def on_open(ws):
            print("Realtime WebSocket Connection Opened")
            ws.send(json.dumps(subscribe_msg))
        
        ws.on_open = on_open
        
        ws.run_forever()
    
    except Exception as e:
        print(f"Realtime listener error: {e}")

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
            'MAE': mean_absolute_error(y_test, y_pred),
        }

        
        # Predict next month's usage
        predicted_usage = model.predict(X)
        
        # Generate recommendations
        recommendations = []
        for idx, row in merged_df.iterrows():
            predicted_monthly_usage = predicted_usage[idx]
            current_quantity = row['quantity']
            
            # Recommend restock with 20% safety buffer
            needed_quantity = predicted_monthly_usage * 1.2  # Total needed with safety buffer
            recommended_restock = max(0, needed_quantity - current_quantity)  # Only restock what's needed
            
            
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

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        # Prepare data
        data = prepare_data()
        
        # Train model and get recommendations
        recommendations, metrics = train_predict_model(data)
        
        return jsonify({
            'recommendations': recommendations,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Start realtime update processing thread
    update_thread = threading.Thread(target=process_realtime_updates, daemon=True)
    update_thread.start()
    
    # Start realtime listener thread
    realtime_thread = threading.Thread(target=start_realtime_listener, daemon=True)
    realtime_thread.start()
    
    # Run Flask app
    app.run(debug=True)