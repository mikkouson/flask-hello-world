from flask import Flask, jsonify
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

load_dotenv()

app = Flask(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/inventory', methods=['GET'])
def get_inventory():
    # Step 1: Fetch inventory data from Supabase
    response = supabase.table('inventory').select('*').execute()
    data = response.data

    # Step 2: Convert the data into a DataFrame
    inventory_df = pd.DataFrame(data)

    # Step 3: Clean and handle invalid values in the 'quantity' column
    inventory_df['quantity'] = inventory_df['quantity'].apply(lambda x: max(x, 0))  # Ensure non-negative quantities
    inventory_df['quantity'] = inventory_df['quantity'].fillna(inventory_df['quantity'].median())  # Fill NaN with median

    # Step 4: Simulate usage data (since real usage data isn't available)
    np.random.seed(42)  # For reproducibility

    # Simulate usage (the number of items used over a period of time, say 30 days)
    inventory_df['usage'] = np.random.normal(30 - 0.1 * inventory_df['quantity'], 5, size=len(inventory_df))

    # Step 5: Feature Engineering
    # Apply log transformation safely (avoid log(0) or negative values)
    inventory_df['log_quantity'] = inventory_df['quantity'].apply(lambda x: np.log(x) if x > 0 else 0)

    # Features (X) and Target (y)
    X = inventory_df[['quantity', 'log_quantity']]  # Features: quantity and log of quantity
    y = inventory_df['usage']  # Target: usage

    # Step 6: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 8: Predict the usage (demand) for the test set
    y_pred = model.predict(X_test)

    # Step 9: Evaluate the model's accuracy using Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error of the model: {mae}")

    # Step 10: Predict demand (usage) for all items in the inventory dataset
    inventory_df['predicted_usage'] = model.predict(X[['quantity', 'log_quantity']])

    # Step 11: Calculate restock amounts based on predicted usage (no fixed target)
    inventory_df['restock_amount'] = np.maximum(0, inventory_df['predicted_usage'] * 1.2 - inventory_df['quantity'])

    # Step 12: Prepare the result to return as JSON, including MAE and branch
    result = {
        'mae': mae,  # Include the Mean Absolute Error in the response
        'inventory': inventory_df[['name', 'branch', 'quantity', 'predicted_usage', 'restock_amount']].to_dict(orient='records')
    }

    # Return the results as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
