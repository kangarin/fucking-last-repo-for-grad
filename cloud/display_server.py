# display_server.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import time
import sys
from pathlib import Path

# Add project root to Python path for importing config
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config import config

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store the latest detection results for each stream
latest_detections = {}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/update_detection', methods=['POST'])
def update_detection():
    """Receive detection results from processing server"""
    try:
        if not request.is_json:
            raise ValueError("Expected JSON request")
            
        data = request.json
        if not data:
            raise ValueError("Empty JSON data")
            
        stream_id = data.get('stream_id')
        if not stream_id:
            raise ValueError("Missing stream_id")
            
        # Add display timestamp and calculate latencies
        timestamps = data.get('timestamps', {})
        timestamps['displayed'] = time.time()
        
        # Calculate latencies
        latencies = {
            'processing_latency': timestamps['processed'] - timestamps['received'],
            'transmission_latency': timestamps['displayed'] - timestamps['processed'],
            'end_to_end_latency': timestamps['displayed'] - timestamps['generated']
        }
        
        # Add latencies to data
        data['latencies'] = latencies
        
        # Update latest detection for this stream
        latest_detections[stream_id] = {
            'frame_id': data.get('frame_id'),
            'timestamps': timestamps,
            'latencies': latencies,
            'image': data.get('image')
        }
        
        # Broadcast the update to all connected clients
        socketio.emit('detection_update', data)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error in update_detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_active_streams')
def get_active_streams():
    """Get list of active streams"""
    return jsonify(list(latest_detections.keys()))

if __name__ == '__main__':
    server_config = config.get_cloud_display_config()
    socketio.run(app, 
                host=server_config['host'], 
                port=server_config['port'], 
                debug=True)