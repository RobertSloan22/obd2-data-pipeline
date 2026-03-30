/**
 * OBD2 feature-to-group mapping for per-group health scoring.
 * Defaults to the original 4 high-level groups, but can be overridden with
 * a JSON file (PID_GROUPS_CONFIG env var, or config/pid-groups.json).
 */

import fs from 'fs';
import path from 'path';

export const DEFAULT_PID_GROUPS = {
  engine: [
    'rpm',
    'speed',
    'engineLoad',
    'absoluteLoad',
    'throttlePosition',
    'relativeThrottlePosition',
    'timingAdvance',
    'runtime',
    'acceleratorPosD',
    'acceleratorPosE',
    'acceleratorPosF'
  ],
  fuel: [
    'fuelLevel',
    'fuelRate',
    'fuelPressure',
    'fuelRailPressure',
    'fuelTrimShortB1',
    'fuelTrimLongB1',
    'fuelTrimShortB2',
    'fuelTrimLongB2',
    'ethanolFuelPercent',
    'maf',
    'map',
    'barometricPressure',
    'vaporPressure'
  ],
  exhaust: [
    'o2B1S1Voltage',
    'o2B1S2Voltage',
    'o2B1S3Voltage',
    'o2B1S4Voltage',
    'o2B2S1Voltage',
    'o2B2S2Voltage',
    'o2B2S3Voltage',
    'o2B2S4Voltage',
    'o2B1S1WR',
    'o2B1S2WR',
    'o2B1S3WR',
    'o2B1S4WR',
    'o2B2S1WR',
    'o2B2S2WR',
    'o2B2S3WR',
    'o2B2S4WR',
    'egrError',
    'commandedEGR',
    'secondaryAirStatus',
    'catalystTempB1S1',
    'catalystTempB1S2',
    'catalystTempB2S1',
    'catalystTempB2S2',
    'shortTermSecondaryO2B1',
    'longTermSecondaryO2B1',
    'shortTermSecondaryO2B2',
    'longTermSecondaryO2B2'
  ],
  cooling: [
    'engineTemp',
    'intakeTemp',
    'ambientTemp',
    'chargeAirCoolerTemp'
  ]
};

function sanitizePidGroups(raw) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }

  const result = {};
  for (const [groupId, features] of Object.entries(raw)) {
    if (!Array.isArray(features)) {
      continue;
    }

    const normalizedFeatures = Array.from(new Set(
      features
        .map((name) => (typeof name === 'string' ? name.trim() : ''))
        .filter(Boolean)
    ));

    if (normalizedFeatures.length > 0) {
      result[groupId] = normalizedFeatures;
    }
  }

  return Object.keys(result).length > 0 ? result : null;
}

function tryLoadGroupConfigFile(configPath) {
  try {
    if (!fs.existsSync(configPath)) {
      return null;
    }

    const parsed = JSON.parse(fs.readFileSync(configPath, 'utf-8'));
    const groups = parsed?.groups ?? parsed;
    const sanitized = sanitizePidGroups(groups);

    if (!sanitized) {
      console.warn(`PID group config found but invalid: ${configPath}`);
      return null;
    }

    return sanitized;
  } catch (error) {
    console.warn(`Failed to parse PID group config at ${configPath}: ${error.message}`);
    return null;
  }
}

function resolvePidGroups() {
  const configuredPath = process.env.PID_GROUPS_CONFIG;
  const defaultPath = path.join(process.cwd(), 'config', 'pid-groups.json');

  const configPath = configuredPath
    ? (path.isAbsolute(configuredPath) ? configuredPath : path.resolve(process.cwd(), configuredPath))
    : defaultPath;

  return tryLoadGroupConfigFile(configPath) || DEFAULT_PID_GROUPS;
}

export const SYSTEMS = resolvePidGroups();
export const SYSTEM_IDS = Object.keys(SYSTEMS);
export const SUPPORTED_PIDS = Array.from(new Set(Object.values(SYSTEMS).flat()));

/**
 * Get feature indices by system for the given feature list
 * @param {string[]} featureNames - Full list of feature names (order matters)
 * @returns {Record<string, number[]>} Map of system ID to array of feature indices
 */
export function getFeatureIndicesBySystem(featureNames) {
  const nameToIndex = new Map(featureNames.map((name, i) => [name, i]));
  const result = {};

  for (const [systemId, systemFeatures] of Object.entries(SYSTEMS)) {
    const indices = [];
    for (const name of systemFeatures) {
      const idx = nameToIndex.get(name);
      if (idx !== undefined) {
        indices.push(idx);
      }
    }
    result[systemId] = indices;
  }

  return result;
}

/**
 * Extract normalized data for a system
 * @param {number[][]} normalizedData - Full normalized data (rows x all features)
 * @param {string[]} featureNames - Full list of feature names
 * @param {object} stats - Full stats { mins, maxs }
 * @param {string} system - System ID (engine, fuel, exhaust, cooling)
 * @returns {{ data: number[][], stats: object }} Subset data and sliced stats for the system
 */
export function extractSystemFeatures(normalizedData, featureNames, stats, system) {
  const indices = getFeatureIndicesBySystem(featureNames)[system];
  if (!indices || indices.length === 0) {
    return { data: [], stats: { mins: [], maxs: [] } };
  }

  const data = normalizedData.map(row => indices.map(i => row[i]));
  const mins = indices.map(i => stats.mins[i]);
  const maxs = indices.map(i => stats.maxs[i]);

  return {
    data,
    stats: { mins, maxs }
  };
}
