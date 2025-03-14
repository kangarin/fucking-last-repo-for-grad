from flask import Flask, render_template, jsonify
import pandas as pd
import os.path
import time

app = Flask(__name__)

# CSV file path
CSV_FILE = '/Users/wenyidai/Development/graduation_projects/model-switch/stats/stats.csv'  # Update this to your actual CSV file path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    try:
        # Check if file exists
        if not os.path.exists(CSV_FILE):
            return jsonify({'error': 'CSV file not found'})
        
        # Read the CSV file
        df = pd.read_csv(CSV_FILE)
        
        # Get the last 60 rows or all if less than 60
        df = df.tail(60)
        
        # Convert timestamp to readable format
        df['formatted_time'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%H:%M:%S')
        
        # Handle cur_model_index which is a categorical value
        if 'cur_model_index' in df.columns:
            # Fixed mapping for model indices: n,s,m,l,x -> 1,2,3,4,5
            model_mapping = {
                'n': 1,
                's': 2,
                'm': 3,
                'l': 4,
                'x': 5
            }
            # Apply the fixed mapping
            df['cur_model_index_numeric'] = df['cur_model_index'].map(model_mapping)
        
        # Prepare data for charts
        result = {
            'timestamps': df['formatted_time'].tolist(),
            'metrics': {}
        }
        
        # All columns except timestamp and formatted_time
        metric_columns = [col for col in df.columns if col not in ['timestamp', 'formatted_time', 'cur_model_index_numeric']]
        
        # Add each metric data
        for column in metric_columns:
            # For cur_model_index, we need special handling
            if column == 'cur_model_index':
                # Add the numeric version for charting
                result['metrics'][column] = df['cur_model_index_numeric'].tolist()
                # Also add the original values as labels
                result['cur_model_index_labels'] = df['cur_model_index'].tolist()
            else:
                result['metrics'][column] = df[column].tolist()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=7654)