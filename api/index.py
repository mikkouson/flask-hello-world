from flask import Flask, jsonify
from supabase import create_client, Client
import websocket
import json
import os
from dotenv import load_dotenv
import threading
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from queue import Queue
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global queue for real-time updates
update_queue = Queue()

class InventoryWebSocket:
    def __init__(self):
        self.ws = None
        self.tables = ["inventory", "items_used"]
        
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            logger.info(f"Received message type: {data.get('event')}")
            
            if data.get('event') == 'INSERT' or data.get('event') == 'UPDATE':
                payload = data.get('payload', {})
                table = data.get('table', '')
                
                if table == 'items_used':
                    self._handle_items_used_update(payload)
                elif table == 'inventory':
                    self._handle_inventory_update(payload)
                
                # Add to update queue for processing
                update_queue.put({
                    'table': table,
                    'event': data.get('event'),
                    'payload': payload,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _handle_items_used_update(self, payload):
        """Handle updates to items_used table"""
        try:
            item_id = payload.get('item_id')
            quantity = payload.get('quantity')
            logger.info(f"New item usage recorded - Item ID: {item_id}, Quantity: {quantity}")
            
            # Trigger prediction update
            self._update_predictions(item_id)
        except Exception as e:
            logger.error(f"Error handling items_used update: {e}")

    def _handle_inventory_update(self, payload):
        """Handle updates to inventory table"""
        try:
            item_id = payload.get('id')
            new_quantity = payload.get('quantity')
            logger.info(f"Inventory updated - Item ID: {item_id}, New Quantity: {new_quantity}")
            
            # Check if below threshold
            self._check_inventory_alerts(item_id, new_quantity)
        except Exception as e:
            logger.error(f"Error handling inventory update: {e}")

    def _update_predictions(self, item_id):
        """Update predictions for specific item"""
        try:
            predictions = predict_next_month_usage([item_id])
            logger.info(f"Updated predictions for item {item_id}: {predictions}")
        except Exception as e:
            logger.error(f"Error updating predictions: {e}")

    def _check_inventory_alerts(self, item_id, quantity):
        """Check if inventory alerts need to be triggered"""
        try:
            predictions = predict_next_month_usage([item_id])
            if item_id in predictions:
                daily_usage = predictions[item_id] / 30
                if daily_usage > 0:
                    days_until_stockout = quantity / daily_usage
                    if days_until_stockout < 7:
                        logger.warning(f"LOW STOCK ALERT - Item {item_id}: {days_until_stockout:.1f} days until stockout")
        except Exception as e:
            logger.error(f"Error checking inventory alerts: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket Connection Closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        logger.info("WebSocket Connection Opened")
        
        # Subscribe to both tables
        for table in self.tables:
            subscribe_msg = {
                "event": "phx_join",
                "topic": f"realtime:public:{table}",
                "payload": {},
                "ref": str(self.tables.index(table) + 1)
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {table} table")

    def start(self):
        """Start WebSocket connection"""
        ws_url = f"{SUPABASE_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/realtime/v1/websocket"
        
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            ws_url,
            header={"apikey": SUPABASE_KEY},
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        self.ws.run_forever()

def fetch_inventory_usage(item_ids=None):
    """Fetch and process historical usage data"""
    try:
        query = supabase.table("items_used").select(
            "item_id", 
            "quantity", 
            "created_at"
        )
        
        if item_ids:
            query = query.in_('item_id', item_ids)
            
        result = query.execute()
        
        df = pd.DataFrame(result.data)
        if df.empty:
            return pd.DataFrame()
            
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['month'] = df['created_at'].dt.to_period('M')
        monthly_usage = df.groupby(['item_id', 'month'])['quantity'].sum().reset_index()
        
        return monthly_usage
    except Exception as e:
        logger.error(f"Error fetching usage data: {e}")
        return pd.DataFrame()

def predict_next_month_usage(item_ids=None):
    """Predict next month's usage for specified items or all items"""
    try:
        monthly_usage = fetch_inventory_usage(item_ids)
        if monthly_usage.empty:
            return {}

        predictions = {}
        for item_id in monthly_usage['item_id'].unique():
            item_data = monthly_usage[monthly_usage['item_id'] == item_id]
            
            if len(item_data) < 3:
                avg_usage = item_data['quantity'].mean()
                predictions[item_id] = round(avg_usage)
                continue
                
            item_data = item_data.set_index('month')['quantity']
            
            try:
                model = ExponentialSmoothing(
                    item_data,
                    trend='add',
                    seasonal='add' if len(item_data) >= 12 else None,
                    seasonal_periods=12 if len(item_data) >= 12 else None
                ).fit()
                
                forecast = model.forecast(steps=1)[0]
                predictions[item_id] = max(round(forecast), 0)
            except Exception as e:
                logger.error(f"Error in prediction for item {item_id}: {e}")
                avg_usage = item_data.mean()
                predictions[item_id] = round(avg_usage)
                
        return predictions
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {}

# API Endpoints
@app.route('/api/inventory/predict', methods=['GET'])
def get_predictions():
    """Get usage predictions and restocking recommendations"""
    try:
        predictions = predict_next_month_usage()
        inventory_data = supabase.table("inventory").select("*").execute()
        
        recommendations = []
        for item in inventory_data.data:
            predicted_usage = predictions.get(item['id'], 0)
            current_quantity = item['quantity']
            
            daily_usage = predicted_usage / 30
            days_until_stockout = round(current_quantity / daily_usage) if daily_usage > 0 else 999
            
            recommendations.append({
                'item_id': item['id'],
                'item_name': item['name'],
                'current_quantity': current_quantity,
                'predicted_monthly_usage': predicted_usage,
                'days_until_stockout': days_until_stockout,
                'status': 'CRITICAL' if days_until_stockout < 7 else 'WARNING' if days_until_stockout < 14 else 'OK'
            })
        
        return jsonify({
            "success": True,
            "data": recommendations,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error in get_predictions: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/inventory/updates', methods=['GET'])
def get_recent_updates():
    """Get recent real-time updates"""
    updates = []
    while not update_queue.empty():
        updates.append(update_queue.get())
    
    return jsonify({
        "success": True,
        "updates": updates,
        "count": len(updates)
    })

if __name__ == '__main__':
    # Start WebSocket in a separate thread
    inventory_ws = InventoryWebSocket()
    websocket_thread = threading.Thread(target=inventory_ws.start, daemon=True)
    websocket_thread.start()
    
    # Run Flask app
    app.run(debug=True)