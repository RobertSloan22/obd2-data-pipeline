#!/usr/bin/env node
import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';
import { MongoClient, ObjectId } from 'mongodb';

import { SYSTEMS } from './systems.js';
import { toFiniteNumber } from './dataPipeline.js';
import { fileSystemIO } from './ioHandler.js';

const SAVE_ROOT = path.join(process.cwd(), 'models');
const CLASSIFIER_DIR = path.join(SAVE_ROOT, 'fault-type-model');
const MODEL_VERSION = 'fault-classifier.v1';

const DEFAULTS = {
  uri: process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017',
  databaseName: process.env.MONGODB_DB || 'automotiveai',
  sessionsCollection: process.env.MONGODB_SESSIONS_COLLECTION || 'diagnosticsessions',
  dataPointsCollection: process.env.MONGODB_COLLECTION || 'obd2datapoints',
  epochs: Number(process.env.TRAIN_CLASSIFIER_EPOCHS || 35),
  batchSize: Number(process.env.TRAIN_CLASSIFIER_BATCH_SIZE || 32),
  minSamplesPerLabel: Number(process.env.TRAIN_CLASSIFIER_MIN_SAMPLES_PER_LABEL || 15),
  maxDataPointsPerSession: Number(process.env.TRAIN_CLASSIFIER_MAX_POINTS_PER_SESSION || 300),
  includeUnconfirmed: process.env.TRAIN_CLASSIFIER_INCLUDE_UNCONFIRMED === 'true',
  verbose: process.env.TRAIN_CLASSIFIER_VERBOSE === '0' ? 0 : 1
};

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = argv[i + 1];
    if (next == null || next.startsWith('--')) {
      if (key === 'includeUnconfirmed') options.includeUnconfirmed = true;
      continue;
    }
    i += 1;
    switch (key) {
      case 'uri':
        options.uri = next;
        break;
      case 'db':
        options.databaseName = next;
        break;
      case 'sessionsCollection':
        options.sessionsCollection = next;
        break;
      case 'collection':
      case 'dataPointsCollection':
        options.dataPointsCollection = next;
        break;
      case 'epochs':
        options.epochs = Number(next);
        break;
      case 'batchSize':
        options.batchSize = Number(next);
        break;
      case 'minSamplesPerLabel':
        options.minSamplesPerLabel = Number(next);
        break;
      case 'maxDataPointsPerSession':
        options.maxDataPointsPerSession = Number(next);
        break;
      default:
        console.warn(`Ignoring unknown option: --${key}`);
        break;
    }
  }
  return options;
}

function normalizeLabel(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_');
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function getFeatureNames() {
  const all = Object.values(SYSTEMS).flat();
  return Array.from(new Set(all));
}

function summarizeSessionPoints(dataPoints, featureNames) {
  const sums = Array(featureNames.length).fill(0);
  const counts = Array(featureNames.length).fill(0);

  dataPoints.forEach((point) => {
    featureNames.forEach((featureName, idx) => {
      const value = toFiniteNumber(point?.[featureName]);
      if (value === null) return;
      sums[idx] += value;
      counts[idx] += 1;
    });
  });

  return featureNames.map((_, idx) => (
    counts[idx] > 0 ? sums[idx] / counts[idx] : null
  ));
}

function computeFeatureStats(rows) {
  const featureCount = rows[0]?.length || 0;
  const mins = Array(featureCount).fill(Infinity);
  const maxs = Array(featureCount).fill(-Infinity);
  const sums = Array(featureCount).fill(0);
  const counts = Array(featureCount).fill(0);

  rows.forEach((row) => {
    row.forEach((value, idx) => {
      if (value == null || !Number.isFinite(value)) return;
      mins[idx] = Math.min(mins[idx], value);
      maxs[idx] = Math.max(maxs[idx], value);
      sums[idx] += value;
      counts[idx] += 1;
    });
  });

  const means = sums.map((sum, idx) => (counts[idx] > 0 ? sum / counts[idx] : 0));
  const safeMins = mins.map((v, idx) => (Number.isFinite(v) ? v : means[idx]));
  const safeMaxs = maxs.map((v, idx) => {
    if (!Number.isFinite(v)) return means[idx] + 1;
    if (v === safeMins[idx]) return v + 1;
    return v;
  });

  return { mins: safeMins, maxs: safeMaxs, means };
}

function normalizeRows(rows, stats) {
  return rows.map((row) => row.map((value, idx) => {
    const min = stats.mins[idx];
    const max = stats.maxs[idx];
    const fallback = stats.means[idx];
    const numeric = Number.isFinite(value) ? value : fallback;
    return (numeric - min) / (max - min);
  }));
}

function buildClassifier(inputDim, classCount) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [inputDim], units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: classCount, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

async function collectLabeledSamples(db, options) {
  const sessions = db.collection(options.sessionsCollection);
  const dataPoints = db.collection(options.dataPointsCollection);
  const featureNames = getFeatureNames();

  const sessionQuery = {
    'metadata.mlLabel.problemType': { $exists: true, $ne: '' }
  };
  if (!options.includeUnconfirmed) {
    sessionQuery['metadata.mlLabel.confirmed'] = true;
  }

  const projection = {
    _id: 1,
    metadata: 1
  };

  const cursor = sessions.find(sessionQuery, { projection });
  const rows = [];
  const labels = [];
  let scannedSessions = 0;

  for await (const session of cursor) {
    scannedSessions += 1;
    const labelRaw = session?.metadata?.mlLabel?.problemType;
    const label = normalizeLabel(labelRaw);
    if (!label) continue;

    const sessionId = session?._id instanceof ObjectId
      ? session._id
      : new ObjectId(String(session?._id));

    const points = await dataPoints.find(
      { sessionId },
      {
        projection: Object.fromEntries(featureNames.map((f) => [f, 1])),
        sort: { timestamp: -1 },
        limit: options.maxDataPointsPerSession
      }
    ).toArray();

    if (!points || points.length === 0) continue;
    const summaryRow = summarizeSessionPoints(points, featureNames);
    if (summaryRow.every((v) => v == null)) continue;

    rows.push(summaryRow);
    labels.push(label);
  }

  return { rows, labels, featureNames, scannedSessions };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  console.log(`Connecting to MongoDB: ${options.uri}`);
  const client = new MongoClient(options.uri);
  await client.connect();

  try {
    const db = client.db(options.databaseName);
    const { rows, labels, featureNames, scannedSessions } = await collectLabeledSamples(db, options);
    if (rows.length === 0) {
      throw new Error('No labeled sessions with usable data points were found.');
    }

    const countsByLabel = labels.reduce((acc, label) => {
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});

    const allowedLabels = Object.entries(countsByLabel)
      .filter(([, count]) => count >= options.minSamplesPerLabel)
      .map(([label]) => label);
    if (allowedLabels.length < 2) {
      throw new Error(
        `Need at least 2 labels with >= ${options.minSamplesPerLabel} samples. Found: ${JSON.stringify(countsByLabel)}`
      );
    }

    const filteredRows = [];
    const filteredLabels = [];
    rows.forEach((row, idx) => {
      if (allowedLabels.includes(labels[idx])) {
        filteredRows.push(row);
        filteredLabels.push(labels[idx]);
      }
    });

    const labelToIndex = Object.fromEntries(allowedLabels.map((label, idx) => [label, idx]));
    const yIndexes = filteredLabels.map((label) => labelToIndex[label]);
    const stats = computeFeatureStats(filteredRows);
    const normalizedRows = normalizeRows(filteredRows, stats);

    const inputTensor = tf.tensor2d(normalizedRows);
    const yTensor = tf.tensor1d(yIndexes, 'int32');
    const yOneHot = tf.oneHot(yTensor, allowedLabels.length);

    const model = buildClassifier(featureNames.length, allowedLabels.length);
    await model.fit(inputTensor, yOneHot, {
      epochs: options.epochs,
      batchSize: Math.min(options.batchSize, Math.max(1, Math.floor(normalizedRows.length / 2))),
      validationSplit: 0.2,
      shuffle: true,
      verbose: options.verbose ? 1 : 0
    });

    ensureDir(CLASSIFIER_DIR);
    await model.save(fileSystemIO(CLASSIFIER_DIR));

    const metadata = {
      modelVersion: MODEL_VERSION,
      trainedAt: new Date().toISOString(),
      featureNames,
      labels: allowedLabels,
      labelMap: labelToIndex,
      labelCounts: allowedLabels.reduce((acc, label) => {
        acc[label] = filteredLabels.filter((x) => x === label).length;
        return acc;
      }, {}),
      stats,
      source: {
        uri: options.uri,
        databaseName: options.databaseName,
        sessionsCollection: options.sessionsCollection,
        dataPointsCollection: options.dataPointsCollection,
        includeUnconfirmed: options.includeUnconfirmed,
        minSamplesPerLabel: options.minSamplesPerLabel,
        maxDataPointsPerSession: options.maxDataPointsPerSession,
        epochs: options.epochs
      },
      sampleCount: filteredRows.length,
      scannedSessions
    };

    fs.writeFileSync(
      path.join(CLASSIFIER_DIR, 'metadata.json'),
      JSON.stringify(metadata, null, 2),
      'utf-8'
    );

    tf.dispose([inputTensor, yTensor, yOneHot]);
    model.dispose();

    console.log(`Saved classifier model to: ${CLASSIFIER_DIR}`);
    console.log(`Training complete. Samples used: ${filteredRows.length}`);
  } finally {
    await client.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

