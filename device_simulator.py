import random
import time
import json
from datetime import datetime
import paho.mqtt.client as mqtt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartHomeDevice:
    def __init__(self, device_id, device_type):
        self.device_id = device_id
        self.device_type = device_type
        self.state = {}
        
    def generate_telemetry(self):
        timestamp = datetime.now().isoformat()
        
        if self.device_type == "temperature_sensor":
            self.state = {
                "temperature": round(random.uniform(20, 30), 2),
                "humidity": round(random.uniform(40, 60), 2)
            }
        elif self.device_type == "light_sensor":
            self.state = {
                "light_level": round(random.uniform(0, 100), 2),
                "is_dark": random.random() > 0.7
            }
        elif self.device_type == "motion_sensor":
            self.state = {
                "motion_detected": random.random() > 0.8,
                "battery_level": round(random.uniform(80, 100), 2)
            }
        elif self.device_type == "smart_lock":
            self.state = {
                "is_locked": random.random() > 0.3,
                "last_access": timestamp
            }
            
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "timestamp": timestamp,
            "data": self.state
        }

def main():
    # Create multiple devices
    devices = [
        SmartHomeDevice("TEMP001", "temperature_sensor"),
        SmartHomeDevice("LIGHT001", "light_sensor"),
        SmartHomeDevice("MOTION001", "motion_sensor"),
        SmartHomeDevice("LOCK001", "smart_lock")
    ]
    
    # MQTT Configuration
    mqtt_broker = "35.198.228.15"  # Change to your VM's IP
    mqtt_port = 1883
    mqtt_topic = "smart_home/telemetry"
    
    # Create MQTT Client
    client = mqtt.Client()
    
    try:
        client.connect(mqtt_broker, mqtt_port, 60)
        client.loop_start()
        
        while True:
            for device in devices:
                telemetry = device.generate_telemetry()
                payload = json.dumps(telemetry)
                client.publish(mqtt_topic, payload)
                logger.info(f"Published data from {device.device_id}: {payload}")
                time.sleep(0.5)  # Simulate different devices sending data
                
    except KeyboardInterrupt:
        logger.info("Stopping device simulation...")
        client.loop_stop()
        client.disconnect()
    except Exception as e:
        logger.error(f"Error in device simulation: {e}")

if __name__ == "__main__":
    main()