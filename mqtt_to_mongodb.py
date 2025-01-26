import paho.mqtt.client as mqtt
import json
from pymongo import MongoClient
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IoTDataBridge:
    def __init__(self):
        # MongoDB configuration
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.db = self.mongo_client['smart_home']
        self.collection = self.db['sensor_data']
        
        # MQTT configuration
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
    def start(self):
        try:
            # Connect to MQTT broker
            self.mqtt_client.connect("localhost", 1883, 60)
            self.mqtt_client.loop_forever()
        except Exception as e:
            logger.error(f"Error in bridge: {e}")
            
    def on_connect(self, client, userdata, flags, rc):
        logger.info("Connected to MQTT broker")
        client.subscribe("smart_home/telemetry")
        
    def on_message(self, client, userdata, msg):
        try:
            # Parse the incoming message
            payload = json.loads(msg.payload.decode())
            
            # Add receipt timestamp
            payload['received_at'] = datetime.now().isoformat()
            
            # Store in MongoDB
            self.collection.insert_one(payload)
            logger.info(f"Stored data: {payload['device_id']}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

if __name__ == "__main__":
    bridge = IoTDataBridge()
    bridge.start()