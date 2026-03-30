# OBD2 Anomaly Detection

TensorFlow.js autoencoder for detecting anomalies in OBD2 vehicle telemetry data (e.g. `autoshop.obd2datapoints.csv`).

## Setup

```bash
npm install
```

## Usage

### 1. Train the model

Train on your CSV data. Provide the path to your OBD2 datapoints CSV:

```bash
npm run train
# or with explicit path:
node src/train.js "C:\Users\rstec\Downloads\autoshop.obd2datapoints.csv"
```

This will:
- Load and normalize the data
- Train an autoencoder on engine metrics (rpm, speed, temps, fuel, O2 sensors, etc.)
- Save the model and metadata under `models/`

### 2. Run anomaly detection

```bash
npm run predict
# or:
node src/predict.js "C:\Users\rstec\Downloads\autoshop.obd2datapoints.csv"
```

Output includes the number of anomalies and the top anomalies by reconstruction error.

## Performance note

Training uses the TensorFlow.js CPU backend by default. For faster training, install `@tensorflow/tfjs-node` and update the imports to use it (the `file://` URL scheme will then work for model save/load).

## How it works

1. **Autoencoder**: Compresses input features into a latent space and reconstructs them. Normal patterns are reconstructed well; anomalies produce higher reconstruction error.
2. **Threshold**: The 99th percentile of training reconstruction errors is used as the anomaly threshold.
3. **Features**: All numeric OBD2 sensor columns are used except metadata (_id, sessionId, timestamps, etc.).

## Project structure

```
obd2-data-pipeline/
├── src/
│   ├── dataLoader.js   # CSV loading, normalization
│   ├── model.js        # Autoencoder definition
│   ├── train.js        # Training script
│   ├── predict.js     # Inference script
│   └── index.js       # Exports
├── models/             # Saved model + metadata (after training)
├── package.json
└── README.md
```

## Customization

- **Encoding dimensions**: Edit `encodingDims` in `src/train.js` (default: `[64, 32, 16]`).
- **Epochs**: Adjust `epochs` in `src/train.js`.
- **Threshold**: Change how the threshold is derived in `src/train.js` (e.g. use `p95` instead of `p99`).
