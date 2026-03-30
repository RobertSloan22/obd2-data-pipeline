#!/usr/bin/env node
/**
 * Run per-system health score prediction directly from MongoDB.
 * Usage:
 *   node src/predictFromMongo.js --uri mongodb://127.0.0.1:27017 --db automotiveai --collection obd2datapoints
 */

import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';
import { MongoClient } from 'mongodb';

import { buildRowFromRecord, isRowUsable, normalizeRow } from './dataPipeline.js';
import { computeReconstructionErrors, reconstructionErrorToHealthScore } from './model.js';
import { fileSystemLoadIO } from './ioHandler.js';
import { SYSTEM_IDS } from './systems.js';

const MODELS_DIR = path.join(process.cwd(), 'models');
const HEALTH_MODEL_BASE = path.join(MODELS_DIR, 'obd2-health-model');

const DEFAULTS = {
  uri: process.env.MONGODB_URI || 'mongodb://127.0.0.1:27017',
  databaseName: process.env.MONGODB_DB || 'automotiveai',
  collectionName: process.env.MONGODB_COLLECTION || 'obd2datapoints',
  batchSize: Number(process.env.PREDICT_BATCH_SIZE || 256),
  findBatchSize: Number(process.env.MONGO_FIND_BATCH_SIZE || 1000),
  minRowCompleteness: Number(process.env.PREDICT_MIN_ROW_COMPLETENESS || 0.5),
  minDataQuality: process.env.PREDICT_MIN_DATA_QUALITY == null
    ? null
    : Number(process.env.PREDICT_MIN_DATA_QUALITY),
  includeInterpolated: process.env.PREDICT_INCLUDE_INTERPOLATED === 'true',
  startDate: process.env.PREDICT_START_DATE || null,
  endDate: process.env.PREDICT_END_DATE || null,
  limit: process.env.PREDICT_LIMIT == null ? null : Number(process.env.PREDICT_LIMIT)
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
      case 'minRowCompleteness':
        options.minRowCompleteness = Number(nextValue);
        break;
      case 'minDataQuality':
        options.minDataQuality = Number(nextValue);
        break;
      case 'start':
        options.startDate = nextValue;
        break;
      case 'end':
        options.endDate = nextValue;
        break;
      case 'limit':
        options.limit = Number(nextValue);
        break;
      default:
        console.warn(`Ignoring unknown option: --${key}`);
        break;
    }
  }

  options.startDate = options.startDate ? new Date(options.startDate) : null;
  options.endDate = options.endDate ? new Date(options.endDate) : null;

  if (options.minDataQuality != null && !Number.isFinite(options.minDataQuality)) {
    options.minDataQuality = null;
  }

  if (options.limit != null && !Number.isFinite(options.limit)) {
    options.limit = null;
  }

  return options;
}

function buildPredictionQuery(options) {
  const query = {};

  if (options.startDate || options.endDate) {
    query.timestamp = {};
    if (options.startDate) {
      query.timestamp.$gte = options.startDate;
    }
    if (options.endDate) {
      query.timestamp.$lt = options.endDate;
    }
  }

  if (options.minDataQuality != null) {
    query.dataQuality = { $gte: options.minDataQuality };
  }

  if (!options.includeInterpolated) {
    query.isInterpolated = { $ne: true };
  }

  return query;
}

function buildProjection(systems) {
  const allFeatures = new Set();
  for (const systemId of SYSTEM_IDS) {
    const featureNames = systems[systemId]?.featureNames || [];
    for (const featureName of featureNames) {
      allFeatures.add(featureName);
    }
  }

  return Object.fromEntries([...allFeatures].map((featureName) => [featureName, 1]));
}

async function scoreChunk(chunk, models, systems, minRowCompleteness) {
  const scored = chunk.map((entry) => ({
    index: entry.index,
    scores: {}
  }));

  const sums = {};
  const counts = {};

  for (const systemId of SYSTEM_IDS) {
    const model = models[systemId];
    const systemConfig = systems[systemId];
    if (!model || !systemConfig) {
      continue;
    }

    const featureNames = systemConfig.featureNames || [];
    const stats = systemConfig.stats;
    if (featureNames.length === 0 || !stats?.mins || !stats?.maxs) {
      continue;
    }

    const rows = [];
    const rowToResultIndex = [];

    for (let i = 0; i < chunk.length; i++) {
      const cleanedRow = buildRowFromRecord(chunk[i].document, featureNames);
      if (!isRowUsable(cleanedRow, minRowCompleteness)) {
        continue;
      }

      rows.push(normalizeRow(cleanedRow, stats));
      rowToResultIndex.push(i);
    }

    if (rows.length === 0) {
      continue;
    }

    const inputTensor = tf.tensor2d(rows);
    const errors = computeReconstructionErrors(model, inputTensor);
    const errorArray = Array.from(await errors.data());
    tf.dispose([inputTensor, errors]);

    for (let i = 0; i < errorArray.length; i++) {
      const error = errorArray[i];
      const resultIndex = rowToResultIndex[i];
      const score = reconstructionErrorToHealthScore(error, systemConfig.threshold);
      scored[resultIndex].scores[systemId] = score;
      sums[systemId] = (sums[systemId] || 0) + score;
      counts[systemId] = (counts[systemId] || 0) + 1;
    }
  }

  return { scored, sums, counts };
}

async function main() {
  const options = parseCliArgs(process.argv.slice(2));

  const metadataPath = path.join(MODELS_DIR, 'metadata.json');
  if (!fs.existsSync(metadataPath)) {
    console.error('No trained model found. Run: npm run train');
    process.exit(1);
  }

  const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
  const { systems } = metadata;
  if (!systems) {
    console.error('Metadata missing systems. Retrain with: npm run train');
    process.exit(1);
  }

  console.log('Loading models...');
  const models = {};
  for (const systemId of SYSTEM_IDS) {
    if (!systems[systemId]) {
      continue;
    }
    const modelPath = path.join(HEALTH_MODEL_BASE, systemId);
    if (!fs.existsSync(path.join(modelPath, 'model.json'))) {
      continue;
    }
    models[systemId] = await tf.loadLayersModel(fileSystemLoadIO(modelPath));
  }

  if (Object.keys(models).length === 0) {
    console.error('No system models found. Run: npm run train');
    process.exit(1);
  }

  const query = buildPredictionQuery(options);
  const projection = buildProjection(systems);

  const client = new MongoClient(options.uri);
  await client.connect();
  const collection = client.db(options.databaseName).collection(options.collectionName);

  const cursor = collection.find(query, {
    projection,
    batchSize: options.findBatchSize
  });

  const preview = [];
  const meanSums = {};
  const meanCounts = {};
  let scanned = 0;
  let scoredSamples = 0;
  let chunk = [];

  try {
    for await (const document of cursor) {
      if (options.limit != null && scanned >= options.limit) {
        break;
      }

      chunk.push({
        index: scanned,
        document
      });
      scanned += 1;

      if (chunk.length < options.batchSize) {
        continue;
      }

      const { scored, sums, counts } = await scoreChunk(
        chunk,
        models,
        systems,
        options.minRowCompleteness
      );

      for (const entry of scored) {
        if (Object.keys(entry.scores).length > 0) {
          scoredSamples += 1;
          if (preview.length < 20) {
            preview.push(entry);
          }
        }
      }
      for (const [systemId, sum] of Object.entries(sums)) {
        meanSums[systemId] = (meanSums[systemId] || 0) + sum;
      }
      for (const [systemId, count] of Object.entries(counts)) {
        meanCounts[systemId] = (meanCounts[systemId] || 0) + count;
      }

      chunk = [];
    }

    if (chunk.length > 0) {
      const { scored, sums, counts } = await scoreChunk(
        chunk,
        models,
        systems,
        options.minRowCompleteness
      );

      for (const entry of scored) {
        if (Object.keys(entry.scores).length > 0) {
          scoredSamples += 1;
          if (preview.length < 20) {
            preview.push(entry);
          }
        }
      }
      for (const [systemId, sum] of Object.entries(sums)) {
        meanSums[systemId] = (meanSums[systemId] || 0) + sum;
      }
      for (const [systemId, count] of Object.entries(counts)) {
        meanCounts[systemId] = (meanCounts[systemId] || 0) + count;
      }
    }
  } finally {
    await client.close();
    Object.values(models).forEach((model) => model.dispose());
  }

  console.log('\n--- Health Scores (Mongo) ---');
  console.log(`Total documents scanned: ${scanned}`);
  console.log(`Total samples scored: ${scoredSamples}`);
  console.log('');

  for (const entry of preview) {
    const parts = Object.entries(entry.scores)
      .map(([key, value]) => `${key}=${value.toFixed(1)}`)
      .join(', ');
    console.log(`  Sample ${entry.index}: ${parts}`);
  }
  if (scoredSamples > preview.length) {
    console.log(`  ... and ${scoredSamples - preview.length} more`);
  }

  console.log('\nMean health by system:');
  for (const systemId of SYSTEM_IDS) {
    const count = meanCounts[systemId] || 0;
    if (count === 0) {
      continue;
    }
    const mean = meanSums[systemId] / count;
    console.log(`  ${systemId}: ${mean.toFixed(1)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
