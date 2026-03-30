/**
 * OBD2 CSV Data Loader and Preprocessor
 * Loads autoshop.obd2datapoints.csv and prepares tensors for anomaly detection
 */

import fs from 'fs';
import { getFeatureIndicesBySystem, SYSTEMS } from './systems.js';
import path from 'path';
import Papa from 'papaparse';
import { toFiniteNumber } from './dataPipeline.js';

/** Columns to exclude from model input (metadata, IDs, timestamps) */
const EXCLUDED_COLUMNS = new Set([
  '_id', 'sessionId', 'timestamp', 'fuelSystemStatus', 'fuelType',
  'dataQuality', 'isInterpolated', '__v', 'createdAt', 'updatedAt'
]);

/**
 * Check if a value is a valid number
 * @param {*} value - Value to check
 * @returns {boolean}
 */
function isNumeric(value) {
  return toFiniteNumber(value) !== null;
}

function parseCsvFile(csvPath) {
  const absPath = path.isAbsolute(csvPath) ? csvPath : path.resolve(process.cwd(), csvPath);
  const fileContent = fs.readFileSync(absPath, 'utf-8');

  const parsed = Papa.parse(fileContent, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: false
  });

  if (parsed.errors.length > 0) {
    console.warn('CSV parse warnings:', parsed.errors.slice(0, 5));
  }

  return parsed;
}

export function loadOBD2Records(csvPath) {
  const parsed = parseCsvFile(csvPath);
  return {
    records: parsed.data,
    featureNames: parsed.meta.fields.filter(col => !EXCLUDED_COLUMNS.has(col))
  };
}

/**
 * Parse CSV and extract numeric feature columns
 * @param {string} csvPath - Path to CSV file
 * @returns {{ data: number[][], featureNames: string[], stats: object }}
 */
export function loadOBD2Data(csvPath) {
  const { records, featureNames: featureColumns } = loadOBD2Records(csvPath);
  const rows = [];

  for (const row of records) {
    const values = [];
    let valid = true;

    for (const col of featureColumns) {
      const val = row[col];
      if (!isNumeric(val)) {
        valid = false;
        break;
      }
      values.push(toFiniteNumber(val));
    }

    if (valid && values.length === featureColumns.length) {
      rows.push(values);
    }
  }

  const stats = computeNormalizationStats(rows);

  return {
    data: rows,
    featureNames: featureColumns,
    stats
  };
}

/**
 * Compute min/max for each feature for normalization
 * @param {number[][]} data - Raw feature matrix
 * @returns {object}
 */
function computeNormalizationStats(data) {
  if (data.length === 0) return { mins: [], maxs: [] };

  const numFeatures = data[0].length;
  const mins = Array(numFeatures).fill(Infinity);
  const maxs = Array(numFeatures).fill(-Infinity);

  for (const row of data) {
    for (let i = 0; i < numFeatures; i++) {
      const v = row[i];
      if (v < mins[i]) mins[i] = v;
      if (v > maxs[i]) maxs[i] = v;
    }
  }

  // Avoid division by zero
  for (let i = 0; i < numFeatures; i++) {
    if (maxs[i] - mins[i] === 0) {
      maxs[i] = mins[i] + 1;
    }
  }

  return { mins, maxs };
}

/**
 * Normalize data to [0, 1] range
 * @param {number[][]} data - Raw data
 * @param {object} stats - { mins, maxs }
 * @returns {number[][]}
 */
export function normalizeData(data, stats) {
  const { mins, maxs } = stats;
  return data.map(row =>
    row.map((v, i) => (v - mins[i]) / (maxs[i] - mins[i]))
  );
}

/**
 * Extract data subset for a system (selected feature columns)
 * @param {number[][]} data - Full raw data (rows x all features)
 * @param {string[]} featureNames - Full list of feature names
 * @param {string} system - System ID (engine, fuel, exhaust, cooling)
 * @param {object} fullStats - Full stats { mins, maxs } for all features
 * @returns {{ data: number[][], featureNames: string[], stats: object }}
 */
export function extractSystemSubset(data, featureNames, system, fullStats) {
  const systemFeatures = SYSTEMS[system];
  if (!systemFeatures) {
    return { data: [], featureNames: [], stats: { mins: [], maxs: [] } };
  }

  const indices = getFeatureIndicesBySystem(featureNames)[system];
  if (!indices || indices.length === 0) {
    return { data: [], featureNames: [], stats: { mins: [], maxs: [] } };
  }

  const subsetData = data.map(row => indices.map(i => row[i]));
  const subsetFeatureNames = indices.map(i => featureNames[i]);
  const mins = indices.map(i => fullStats.mins[i]);
  const maxs = indices.map(i => fullStats.maxs[i]);

  // Avoid division by zero
  const stats = { mins: [...mins], maxs: [...maxs] };
  for (let i = 0; i < stats.mins.length; i++) {
    if (stats.maxs[i] - stats.mins[i] === 0) {
      stats.maxs[i] = stats.mins[i] + 1;
    }
  }

  return {
    data: subsetData,
    featureNames: subsetFeatureNames,
    stats
  };
}
