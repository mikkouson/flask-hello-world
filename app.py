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

        # Fetch inventory logs to get last restock dates
        inventory_logs = supabase.table("inventory_logs").select("*").execute()
        logs_df = pd.DataFrame(inventory_logs.data)
        
        # Get the latest restock date for each item
        latest_restock_dates = logs_df.sort_values('created_at').groupby('item_id')['created_at'].last()

        # Fetch usage data
        usage_data = supabase.table("items_used").select("*").execute()
        usage_df = pd.DataFrame(usage_data.data)

        # Filter usage data based on restock dates
        filtered_usage = []
        for item_id in usage_df['item_id'].unique():
            item_usage = usage_df[usage_df['item_id'] == item_id].copy()
            if item_id in latest_restock_dates.index:
                last_restock = pd.to_datetime(latest_restock_dates[item_id])
                item_usage = item_usage[pd.to_datetime(item_usage['created_at']) > last_restock]
            filtered_usage.append(item_usage)
        
        filtered_usage_df = pd.concat(filtered_usage, ignore_index=True)

        # Create pivot table with filtered data
        usage_pivot = filtered_usage_df.pivot_table(
            index='item_id', 
            columns='created_at', 
            values='quantity', 
            aggfunc='sum'
        ).reset_index()

        # Fill missing dates with 0
        usage_pivot = usage_pivot.fillna(0)

        # Merge inventory with filtered usage data
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
        
        # Filter out items with insufficient data
        valid_items_mask = (merged_df[date_columns] > 0).sum(axis=1) >= 3
        filtered_df = merged_df[valid_items_mask].copy()
        
        # Prepare features matrix for valid items
        X = filtered_df[date_columns]
        
        # Prepare Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Total usage as target
        y = X.sum(axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict next month's usage for valid items
        predicted_usage = model.predict(X)
        
        # Generate recommendations
        recommendations = []
        for idx, row in filtered_df.iterrows():
            current_quantity = row['quantity']
            predicted_monthly_usage = int(predicted_usage[filtered_df.index.get_loc(idx)])
            
            # Recommend restock with 20% safety buffer
            needed_quantity = predicted_monthly_usage * 1.2  # Total needed with safety buffer
            recommended_restock = max(0, round(needed_quantity - current_quantity))  # Only restock what's needed
            
            recommendations.append({
                'item_id': row['id'],
                'current_quantity': current_quantity,
                'predicted_monthly_usage': predicted_monthly_usage,
                'recommended_restock': recommended_restock
            })

        # Add items with insufficient data to recommendations with zero predictions
        for idx, row in merged_df[~valid_items_mask].iterrows():
            recommendations.append({
                'item_id': row['id'],
                'current_quantity': row['quantity'],
                'predicted_monthly_usage': 0,
                'recommended_restock': 0
            })

        return recommendations, {}

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