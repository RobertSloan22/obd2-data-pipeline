#!/usr/bin/env node
/**
 * Run anomaly detection on OBD2 data using trained model
 * Usage: node src/predict.js [path/to/data.csv]
 */

import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';

import { loadOBD2Data, normalizeData } from './dataLoader.js';
import { computeReconstructionErrors } from './model.js';
import { fileSystemLoadIO } from './ioHandler.js';

const DEFAULT_CSV_PATH = path.join(process.cwd(), '..', 'Downloads', 'autoshop.obd2datapoints.csv');
const MODELS_DIR = path.join(process.cwd(), 'models');

async function main() {
  const csvPath = process.argv[2] || DEFAULT_CSV_PATH;

  const modelPath = path.join(MODELS_DIR, 'obd2-anomaly-model');
  const metadataPath = path.join(MODELS_DIR, 'metadata.json');

  if (!fs.existsSync(metadataPath)) {
    console.error('No trained model found. Run: npm run train');
    process.exit(1);
  }

  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
  const { stats, threshold, systems } = metadata;

  if (systems && !threshold) {
    console.error(
      'This project uses per-system health models. Use: npm run predict:health'
    );
    process.exit(1);
  }

  if (!stats || threshold == null) {
    console.error('No trained anomaly model found. Run: npm run train');
    process.exit(1);
  }

  if (!fs.existsSync(path.join(modelPath, 'model.json'))) {
    console.error(
      'Anomaly model not found. Train with the legacy setup or use: npm run predict:health'
    );
    process.exit(1);
  }

  console.log('Loading model from:', modelPath);
  const model = await tf.loadLayersModel(fileSystemLoadIO(modelPath));

  console.log('Loading data from:', csvPath);
  const { data, featureNames } = loadOBD2Data(csvPath);
  const normalized = normalizeData(data, stats);

  const inputTensor = tf.tensor2d(normalized);
  const errors = computeReconstructionErrors(model, inputTensor);
  const errorsArray = await errors.dataSync();

  const anomalies = [];
  for (let i = 0; i < errorsArray.length; i++) {
    if (errorsArray[i] > threshold) {
      anomalies.push({
        index: i,
        error: errorsArray[i],
        sample: data[i]
      });
    }
  }

  console.log('\n--- Results ---');
  console.log(`Total samples: ${data.length}`);
  console.log(`Anomalies detected: ${anomalies.length} (threshold: ${threshold.toFixed(6)})`);

  if (anomalies.length > 0) {
    console.log('\nTop 10 anomalies (by reconstruction error):');
    const top = anomalies
      .sort((a, b) => b.error - a.error)
      .slice(0, 10);

    top.forEach((a, i) => {
      const samplePreview = featureNames.slice(0, 5).map((name, j) =>
        `${name}=${a.sample[j]?.toFixed(2) ?? '?'}`
      ).join(', ');
      console.log(`  ${i + 1}. idx=${a.index} error=${a.error.toFixed(6)} | ${samplePreview}...`);
    });
  }

  tf.dispose([inputTensor, errors]);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
