#!/usr/bin/env node
/**
 * Evaluate anomaly detection model - show error distribution
 * Usage: node src/evaluate.js [path/to/data.csv]
 */

import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';

import { loadOBD2Data, normalizeData } from './dataLoader.js';
import { computeReconstructionErrors } from './model.js';
import { fileSystemLoadIO } from './ioHandler.js';

const DEFAULT_CSV_PATH = path.join(process.cwd(), '..', 'Downloads', 'autoshop.obd2datapoints.csv');
const MODELS_DIR = path.join(process.cwd(), 'models');

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length * p / 100)];
}

async function main() {
  const csvPath = process.argv[2] || DEFAULT_CSV_PATH;

  const metadataPath = path.join(MODELS_DIR, 'metadata.json');
  if (!fs.existsSync(metadataPath)) {
    console.error('No trained model found. Run: npm run train');
    process.exit(1);
  }

  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
  const model = await tf.loadLayersModel(fileSystemLoadIO(path.join(MODELS_DIR, 'obd2-anomaly-model')));

  const { data, featureNames } = loadOBD2Data(csvPath);
  const normalized = normalizeData(data, metadata.stats);

  const inputTensor = tf.tensor2d(normalized);
  const errors = computeReconstructionErrors(model, inputTensor);
  const errorsArray = await errors.dataSync();

  const mean = errorsArray.reduce((a, b) => a + b, 0) / errorsArray.length;
  const variance = errorsArray.reduce((s, e) => s + (e - mean) ** 2, 0) / errorsArray.length;
  const std = Math.sqrt(variance);

  console.log('\n--- Reconstruction Error Distribution ---');
  console.log(`Samples: ${errorsArray.length}`);
  console.log(`Mean:   ${mean.toFixed(6)}`);
  console.log(`Std:    ${std.toFixed(6)}`);
  console.log(`Min:    ${Math.min(...errorsArray).toFixed(6)}`);
  console.log(`Max:    ${Math.max(...errorsArray).toFixed(6)}`);
  console.log(`\nPercentiles:`);
  [50, 90, 95, 99, 99.5].forEach(p => {
    console.log(`  ${p}%: ${percentile(errorsArray, p).toFixed(6)}`);
  });
  console.log(`\nThreshold (from training): ${metadata.threshold.toFixed(6)}`);
  const anomalyCount = errorsArray.filter(e => e > metadata.threshold).length;
  console.log(`Anomalies at this threshold: ${anomalyCount} (${(100 * anomalyCount / errorsArray.length).toFixed(1)}%)`);

  tf.dispose([inputTensor, errors]);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
