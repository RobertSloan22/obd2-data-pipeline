#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';

import { toFiniteNumber } from './dataPipeline.js';
import { fileSystemLoadIO } from './ioHandler.js';

const MODELS_DIR = path.join(process.cwd(), 'models');
const CLASSIFIER_DIR = path.join(MODELS_DIR, 'fault-type-model');
const METADATA_PATH = path.join(CLASSIFIER_DIR, 'metadata.json');

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;
    const key = raw.slice(2);
    const next = argv[i + 1];
    if (next == null || next.startsWith('--')) {
      args[key] = true;
      continue;
    }
    i += 1;
    args[key] = next;
  }
  return args;
}

function normalizeInputRow(input, featureNames) {
  return featureNames.map((featureName) => toFiniteNumber(input?.[featureName]));
}

function normalizeRowWithStats(row, stats) {
  return row.map((value, idx) => {
    const min = stats.mins[idx];
    const max = stats.maxs[idx];
    const fallback = stats.means?.[idx] ?? min;
    const numeric = Number.isFinite(value) ? value : fallback;
    return (numeric - min) / (max - min);
  });
}

function getTopPrediction(probabilities, labels) {
  let topIndex = 0;
  let topValue = -Infinity;
  probabilities.forEach((value, idx) => {
    if (value > topValue) {
      topValue = value;
      topIndex = idx;
    }
  });
  return {
    problemType: labels[topIndex] || 'unknown',
    confidence: Number(topValue)
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const rowJson = args.json || args.row || null;
  if (!rowJson) {
    console.error('Missing --json payload. Example: node src/predictFaultType.js --json "{\\"rpm\\":1300,\\"maf\\":7.1}"');
    process.exit(1);
  }

  if (!fs.existsSync(METADATA_PATH)) {
    console.error('Missing classifier metadata. Run: npm run train:fault-classifier');
    process.exit(1);
  }
  if (!fs.existsSync(path.join(CLASSIFIER_DIR, 'model.json'))) {
    console.error('Missing classifier model. Run: npm run train:fault-classifier');
    process.exit(1);
  }

  const metadata = JSON.parse(fs.readFileSync(METADATA_PATH, 'utf-8'));
  const inputObj = JSON.parse(String(rowJson));
  const labels = Array.isArray(metadata.labels) ? metadata.labels : Object.keys(metadata.labelMap || {});
  if (!Array.isArray(metadata.featureNames) || metadata.featureNames.length === 0) {
    throw new Error('metadata.featureNames is missing or empty.');
  }
  if (!metadata.stats?.mins || !metadata.stats?.maxs) {
    throw new Error('metadata.stats is missing.');
  }

  const model = await tf.loadLayersModel(fileSystemLoadIO(CLASSIFIER_DIR));
  const rawRow = normalizeInputRow(inputObj, metadata.featureNames);
  const normalizedRow = normalizeRowWithStats(rawRow, metadata.stats);
  const inputTensor = tf.tensor2d([normalizedRow]);
  const output = model.predict(inputTensor);
  const probs = Array.from(await output.data());
  const top = getTopPrediction(probs, labels);
  const probabilities = labels.reduce((acc, label, idx) => {
    acc[label] = Number(probs[idx] || 0);
    return acc;
  }, {});

  const response = {
    success: true,
    modelVersion: metadata.modelVersion || 'unknown',
    prediction: {
      problemType: top.problemType,
      confidence: top.confidence,
      probabilities
    },
    timestamp: new Date().toISOString()
  };

  const shouldPretty = Boolean(args.pretty);
  console.log(JSON.stringify(response, null, shouldPretty ? 2 : 0));

  tf.dispose([inputTensor, output]);
  model.dispose();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

