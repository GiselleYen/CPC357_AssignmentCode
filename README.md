# Project Overview

This repository contains scripts to simulate a device, process data through MQTT, and utilize machine learning for data analysis and model deployment to Google Cloud's Vertex AI.

### Local Environment

1. Clone the repository

2. Open the project in VS Code

## Usage

### Simulate Device Data

1. Open the VS Code terminal.

2. Run the device simulator script:

   ```bash
   python device_simulator.py
   ```

### Process Data on GCP VM

1. SSH into the GCP VM:

2. Run the MQTT-to-mongodb script:

   ```bash
   python3 mqtt_to_mongodb.py
   ```

   This script receives data from the MQTT broker and stores it in the MongoDB database.

### Train and Deploy Machine Learning Model

1. Execute the ML analyzer script:
   ```bash
   python3 ml_analyzer.py
   ```
---

## Files Overview

- **device\_simulator.py**: Simulates device data and publishes it to an MQTT broker.
- **mqtt\_to\_mongodb.py**: Processes MQTT messages and stores data into the MongoDB database.
- **ml\_analyzer.py**: Trains and tests a machine learning model, then deploys it to Vertex AI.
