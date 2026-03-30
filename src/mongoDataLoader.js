import { SYSTEMS } from './systems.js';
import {
  toFiniteNumber,
  buildRowFromRecord,
  isRowUsable,
  normalizeRow
} from './dataPipeline.js';

const DEFAULT_FIND_BATCH_SIZE = 1000;

function buildProjection(featureNames) {
  return Object.fromEntries(featureNames.map((featureName) => [featureName, 1]));
}

function getSystemFeatures(systemId) {
  return SYSTEMS[systemId] || [];
}

function buildCursor(collection, query, projection, findBatchSize) {
  return collection.find(query, {
    projection,
    batchSize: findBatchSize
  });
}

function redactMongoUri(uri) {
  try {
    const parsed = new URL(uri);
    if (parsed.username || parsed.password) {
      parsed.username = '***';
      parsed.password = '***';
    }
    return parsed.toString();
  } catch {
    return uri;
  }
}

export function buildTrainingQuery(options = {}) {
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

  if (
    options.minDataQuality != null &&
    Number.isFinite(Number(options.minDataQuality))
  ) {
    query.dataQuality = { $gte: Number(options.minDataQuality) };
  }

  // Strict `false` excludes documents where the field is missing; `$ne: true`
  // still treats missing fields as "not interpolated" in MongoDB.
  if (!options.includeInterpolated) {
    query.isInterpolated = { $ne: true };
  }

  return query;
}

export async function scanSystemData(collection, systemId, options = {}) {
  const candidateFeatures = getSystemFeatures(systemId);
  const query = buildTrainingQuery(options);
  const findBatchSize = options.findBatchSize || DEFAULT_FIND_BATCH_SIZE;

  if (candidateFeatures.length === 0) {
    return {
      systemId,
      query,
      featureNames: [],
      stats: { mins: [], maxs: [], means: [] },
      scannedDocuments: 0,
      featureCoverage: {}
    };
  }

  const counts = Array(candidateFeatures.length).fill(0);
  const sums = Array(candidateFeatures.length).fill(0);
  const mins = Array(candidateFeatures.length).fill(Infinity);
  const maxs = Array(candidateFeatures.length).fill(-Infinity);
  let scannedDocuments = 0;

  const cursor = buildCursor(collection, query, buildProjection(candidateFeatures), findBatchSize);

  for await (const document of cursor) {
    scannedDocuments += 1;

    candidateFeatures.forEach((featureName, index) => {
      const value = toFiniteNumber(document[featureName]);
      if (value === null) {
        return;
      }

      counts[index] += 1;
      sums[index] += value;
      mins[index] = Math.min(mins[index], value);
      maxs[index] = Math.max(maxs[index], value);
    });
  }

  const minFeaturePresence = options.minFeaturePresence ?? 0.4;
  const minFeatureSamples = options.minFeatureSamples ?? 10;
  const selectedIndices = candidateFeatures
    .map((featureName, index) => ({ featureName, index }))
    .filter(({ index }) => {
      if (counts[index] < minFeatureSamples) {
        return false;
      }

      if (scannedDocuments === 0) {
        return false;
      }

      const coverage = counts[index] / scannedDocuments;
      if (coverage < minFeaturePresence) {
        return false;
      }

      return maxs[index] > mins[index];
    });

  const featureNames = selectedIndices.map(({ featureName }) => featureName);
  const stats = {
    mins: selectedIndices.map(({ index }) => mins[index]),
    maxs: selectedIndices.map(({ index }) => {
      const value = maxs[index];
      return value === mins[index] ? value + 1 : value;
    }),
    means: selectedIndices.map(({ index }) => sums[index] / counts[index])
  };

  const featureCoverage = Object.fromEntries(
    candidateFeatures.map((featureName, index) => [
      featureName,
      scannedDocuments === 0 ? 0 : counts[index] / scannedDocuments
    ])
  );

  return {
    systemId,
    query,
    featureNames,
    stats,
    scannedDocuments,
    featureCoverage
  };
}

export async function countUsableRows(collection, featureNames, options = {}) {
  if (featureNames.length === 0) {
    return 0;
  }

  const query = buildTrainingQuery(options);
  const findBatchSize = options.findBatchSize || DEFAULT_FIND_BATCH_SIZE;
  const minRowCompleteness = options.minRowCompleteness ?? 0.8;
  const cursor = buildCursor(collection, query, buildProjection(featureNames), findBatchSize);
  let usableRows = 0;

  for await (const document of cursor) {
    const row = buildRowFromRecord(document, featureNames);
    if (!isRowUsable(row, minRowCompleteness)) {
      continue;
    }

    usableRows += 1;
  }

  return usableRows;
}

export async function* streamNormalizedSystemBatches(collection, featureNames, stats, options = {}) {
  if (featureNames.length === 0) {
    return;
  }

  const query = buildTrainingQuery(options);
  const findBatchSize = options.findBatchSize || DEFAULT_FIND_BATCH_SIZE;
  const minRowCompleteness = options.minRowCompleteness ?? 0.8;
  const batchSize = options.batchSize ?? 256;
  const cursor = buildCursor(collection, query, buildProjection(featureNames), findBatchSize);
  let batch = [];

  for await (const document of cursor) {
    const row = buildRowFromRecord(document, featureNames);
    if (!isRowUsable(row, minRowCompleteness)) {
      continue;
    }

    batch.push(normalizeRow(row, stats));
    if (batch.length >= batchSize) {
      yield batch;
      batch = [];
    }
  }

  if (batch.length > 0) {
    yield batch;
  }
}

export function summarizeTrainingSource({ uri, databaseName, collectionName, ...options }) {
  return {
    uri: redactMongoUri(uri),
    databaseName,
    collectionName,
    ...options
  };
}
