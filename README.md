# CoralCom - Telemetry Relay and Logger

CoralCom acts as a relay and logging service for telemetry data received from the AutoPi rover. It is designed to retransmit the data to Mission Control and log telemetry for analysis and debugging.

## Features

- **Telemetry Reception:** Listens for telemetry data transmitted from the AutoPi rover.
- **Data Relay:** Forwards telemetry data to Mission Control using TCP.
- **Logging:** Logs telemetry data for historical analysis and debugging purposes.
- **Link Monitoring:** Monitors and reports connection statuses between AutoPi, CoralCom, and Mission Control.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Files

### `retransmission.py`
Handles the core functionality of receiving, logging, and retransmitting telemetry data.

### Logging Directory
All logs are stored in the `logs/` directory, with filenames generated based on the current date and time.

## Usage

1. **Start CoralCom**
   Run the `retransmission.py` script to begin listening for telemetry data from AutoPi and forwarding it to Mission Control.

   ```bash
   python retransmission.py
   ```

2. **Logging**
   Logs are automatically created in the `logs/` directory, storing all received telemetry data for later inspection.

3. **Configuration**
   The script can be configured by modifying the following variables in `retransmission.py`:

   - `UDP_IP` and `UDP_PORT`: Specifies the IP address and port to listen for telemetry from AutoPi.
   - `TCP_IP` and `TCP_PORT`: Specifies the IP address and port for forwarding data to Mission Control.

## How It Works

1. **Receive Telemetry**
   CoralCom listens for incoming UDP telemetry data from AutoPi.

2. **Log Data**
   Each received telemetry packet is logged with a timestamp to ensure traceability.

3. **Relay Data**
   CoralCom retransmits the received telemetry data to Mission Control using a reliable TCP connection.

4. **Link Monitoring**
   Regularly checks and reports the connection statuses between the various components.

## Example Output

### Logs
Logs are stored in JSON format, with each line representing a single telemetry packet:
```json
{
    "timestamp": "2025-01-26T12:00:00",
    "position": {"x": 5.0, "y": 10.0},
    "heading": 90,
    "battery_level": 80.5,
    "ultrasound_distance": 3.2,
    "system_state": {
        "cpu_usage": 45.0,
        "memory_available": 2500,
        "disk_usage": 70.0
    }
}
```

### Console Output
During operation, the script outputs status updates to the console:
```bash
Listening for telemetry on UDP 127.0.0.1:50055
Relaying telemetry to Mission Control on TCP 192.168.0.100:60000
Telemetry received and logged.
Connection to Mission Control: 🟢
Connection to AutoPi: 🟢
```

## Troubleshooting

### "Connection to AutoPi: 🔴"
- Check that AutoPi is running and transmitting telemetry to the correct IP and port.
- Verify network connectivity between AutoPi and CoralCom.

### "Connection to Mission Control: 🔴"
- Ensure Mission Control is running and listening on the correct TCP IP and port.
- Check network connectivity between CoralCom and Mission Control.

### Logs Not Being Generated
- Verify that the `logs/` directory exists and is writable.
- Check for file permission issues.

## Future Enhancements

- **Redundancy:** Implement a fallback mechanism for retransmission to ensure data is not lost during network interruptions.

