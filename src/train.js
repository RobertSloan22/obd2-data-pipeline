#!/usr/bin/env node
/**
 * Train OBD2 per-system health score models
 * Trains one autoencoder per system (engine, fuel, exhaust, cooling)
 * Usage:
 *   node src/train.js --uri mongodb://127.0.0.1:27017 --db automotiveai --collection obd2datapoints
 */

import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';
import { MongoClient } from 'mongodb';

import {
  buildTrainingQuery,
  scanSystemData,
  countUsableRows,
  streamNormalizedSystemBatches,
  summarizeTrainingSource
} from './mongoDataLoader.js';
import { buildAutoencoder, trainModelOnBatches, computeReconstructionErrors } from './model.js';
import { fileSystemIO } from './ioHandler.js';
import { SYSTEM_IDS } from './systems.js';

const SAVE_DIR = path.join(process.cwd(), 'models');
const HEALTH_MODEL_BASE = path.join(SAVE_DIR, 'obd2-health-model');

const DEFAULTS = {
  uri: process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017',
  databaseName: process.env.MONGODB_DB || 'automotiveai',
  collectionName: process.env.MONGODB_COLLECTION || 'obd2datapoints',
  batchSize: Number(process.env.TRAIN_BATCH_SIZE || 256),
  findBatchSize: Number(process.env.MONGO_FIND_BATCH_SIZE || 1000),
  epochs: Number(process.env.TRAIN_EPOCHS || 50),
  minDataQuality: (() => {
    const value = process.env.TRAIN_MIN_DATA_QUALITY;
    if (value == null || value === '') {
      return null;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  })(),
  includeInterpolated: process.env.TRAIN_INCLUDE_INTERPOLATED === 'true',
  minFeaturePresence: Number(process.env.TRAIN_MIN_FEATURE_PRESENCE || 0.4),
  minFeatureSamples: Number(process.env.TRAIN_MIN_FEATURE_SAMPLES || 25),
  minRowCompleteness: Number(process.env.TRAIN_MIN_ROW_COMPLETENESS || 0.8),
  minSamples: Number(process.env.TRAIN_MIN_SAMPLES || 50),
  startDate: process.env.TRAIN_START_DATE || null,
  endDate: process.env.TRAIN_END_DATE || null,
  verbose: process.env.TRAIN_VERBOSE === '0' ? 0 : 1
};

function parseCliArgs(argv) {
  const options = { ...DEFAULTS };

  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (!arg.startsWith('--')) {
      continue;
    }

    const key = arg.slice(2);
    const nextValue = argv[index + 1];

    if (nextValue == null || nextValue.startsWith('--')) {
      if (key === 'includeInterpolated') {
        options.includeInterpolated = true;
      } else if (key === 'quiet') {
        options.verbose = 0;
      }
      continue;
    }

    index += 1;

    switch (key) {
      case 'uri':
        options.uri = nextValue;
        break;
      case 'db':
        options.databaseName = nextValue;
        break;
      case 'collection':
        options.collectionName = nextValue;
        break;
      case 'batchSize':
        options.batchSize = Number(nextValue);
        break;
      case 'findBatchSize':
        options.findBatchSize = Number(nextValue);
        break;
      case 'epochs':
        options.epochs = Number(nextValue);
        break;
      case 'minDataQuality':
        options.minDataQuality = Number(nextValue);
        break;
      case 'minFeaturePresence':
        options.minFeaturePresence = Number(nextValue);
        break;
      case 'minFeatureSamples':
        options.minFeatureSamples = Number(nextValue);
        break;
      case 'minRowCompleteness':
        options.minRowCompleteness = Number(nextValue);
        break;
      case 'minSamples':
        options.minSamples = Number(nextValue);
        break;
      case 'start':
        options.startDate = nextValue;
        break;
      case 'end':
        options.endDate = nextValue;
        break;
      default:
        console.warn(`Ignoring unknown option: --${key}`);
        break;
    }
  }

  options.startDate = options.startDate ? new Date(options.startDate) : null;
  options.endDate = options.endDate ? new Date(options.endDate) : null;

  return options;
}

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function getEncodingDims(inputDim) {
  if (inputDim <= 8) {
    return [16, 8];
  }

  if (inputDim <= 16) {
    return [32, 16];
  }

  return [64, 32, 16];
}

async function collectErrorStats(model, createBatchIterator) {
  const errorsArray = [];

  for await (const batch of createBatchIterator()) {
    const batchTensor = tf.tensor2d(batch);
    const errors = computeReconstructionErrors(model, batchTensor);
    const batchErrors = Array.from(await errors.data());
    errorsArray.push(...batchErrors);
    tf.dispose([batchTensor, errors]);
  }

  if (errorsArray.length === 0) {
    throw new Error('No usable rows were available for reconstruction error analysis.');
  }

  const sorted = [...errorsArray].sort((a, b) => a - b);
  const mean = errorsArray.reduce((sum, value) => sum + value, 0) / errorsArray.length;
  const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
  const p99 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.99))];

  return {
    mean,
    p95,
    p99,
    sampleCount: errorsArray.length
  };
}

async function trainSystemModel(collection, systemId, options) {
  console.log(`\n--- ${systemId.toUpperCase()} ---`);

  const scanResult = await scanSystemData(collection, systemId, options);
  console.log(
    `Scanned ${scanResult.scannedDocuments} documents, selected ${scanResult.featureNames.length} features`
  );

  if (scanResult.featureNames.length === 0) {
    console.warn(`Skipping ${systemId}: no sufficiently populated numeric features found`);
    return null;
  }

  const usableRows = await countUsableRows(collection, scanResult.featureNames, options);
  console.log(`Usable rows after completeness filter: ${usableRows}`);

  if (usableRows < options.minSamples) {
    console.warn(
      `Skipping ${systemId}: need at least ${options.minSamples} usable rows, found ${usableRows}`
    );
    return null;
  }

  const inputDim = scanResult.featureNames.length;
  const model = buildAutoencoder(inputDim, {
    encodingDims: getEncodingDims(inputDim),
    activation: 'relu'
  });

  model.summary();

  await trainModelOnBatches(
    model,
    () => streamNormalizedSystemBatches(
      collection,
      scanResult.featureNames,
      scanResult.stats,
      options
    ),
    {
      epochs: options.epochs,
      verbose: options.verbose
    }
  );

  const errorStats = await collectErrorStats(
    model,
    () => streamNormalizedSystemBatches(
      collection,
      scanResult.featureNames,
      scanResult.stats,
      options
    )
  );

  console.log(
    `Mean error: ${errorStats.mean.toFixed(6)}, p99: ${errorStats.p99.toFixed(6)}`
  );

  return {
    model,
    metadata: {
      featureNames: scanResult.featureNames,
      stats: scanResult.stats,
      threshold: errorStats.p99,
      meanError: errorStats.mean,
      p95Error: errorStats.p95,
      p99Error: errorStats.p99,
      inputDim,
      scannedDocuments: scanResult.scannedDocuments,
      usableRows,
      featureCoverage: scanResult.featureCoverage
    }
  };
}

async function main() {
  const options = parseCliArgs(process.argv.slice(2));
  console.log('Connecting to MongoDB:', options.uri);
  console.log(`Database: ${options.databaseName}, collection: ${options.collectionName}`);

  const client = new MongoClient(options.uri);
  await client.connect();

  ensureDir(SAVE_DIR);
  ensureDir(HEALTH_MODEL_BASE);

  const collection = client.db(options.databaseName).collection(options.collectionName);

  const trainingFilter = buildTrainingQuery(options);
  const totalDocs = await collection.countDocuments({});
  const matchingDocs = await collection.countDocuments(trainingFilter);
  console.log(`Documents in collection: ${totalDocs}, matching training filter: ${matchingDocs}`);
  if (totalDocs === 0) {
    console.warn(
      'No documents in this collection. Check --db / --collection and that Mongo is the instance with your data (e.g. remote host:port).'
    );
  } else if (matchingDocs === 0) {
    console.warn(
      'Filters match 0 documents. Relax filters: add --includeInterpolated, drop --start/--end, or set --minDataQuality only when you need it.'
    );
  }

  const systemsMetadata = {};
  try {
    for (const systemId of SYSTEM_IDS) {
      const trainedSystem = await trainSystemModel(collection, systemId, options);
      if (!trainedSystem) {
        continue;
      }

      const modelPath = path.join(HEALTH_MODEL_BASE, systemId);
      await trainedSystem.model.save(fileSystemIO(modelPath));
      systemsMetadata[systemId] = trainedSystem.metadata;
      trainedSystem.model.dispose();
    }
  } finally {
    await client.close();
  }

  if (Object.keys(systemsMetadata).length === 0) {
    console.error('Training did not produce any system models.');
    process.exit(1);
  }

  const metadata = {
    systems: systemsMetadata,
    source: summarizeTrainingSource({
      uri: options.uri,
      databaseName: options.databaseName,
      collectionName: options.collectionName,
      startDate: options.startDate?.toISOString() || null,
      endDate: options.endDate?.toISOString() || null,
      minDataQuality: options.minDataQuality,
      includeInterpolated: options.includeInterpolated,
      batchSize: options.batchSize,
      findBatchSize: options.findBatchSize,
      minFeaturePresence: options.minFeaturePresence,
      minFeatureSamples: options.minFeatureSamples,
      minRowCompleteness: options.minRowCompleteness,
      minSamples: options.minSamples,
      epochs: options.epochs
    }),
    trainedAt: new Date().toISOString()
  };

  fs.writeFileSync(
    path.join(SAVE_DIR, 'metadata.json'),
    JSON.stringify(metadata, null, 2)
  );

  console.log('\nModels saved to:', HEALTH_MODEL_BASE);
  console.log('Metadata saved to:', path.join(SAVE_DIR, 'metadata.json'));
  console.log('\nDone.');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
