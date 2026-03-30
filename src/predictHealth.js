#!/usr/bin/env node
/**
 * Run per-system health score prediction on OBD2 data
 * Output: { engine, fuel, exhaust, cooling } health scores (0-100) per sample
 * Usage: node src/predictHealth.js [path/to/data.csv]
 */

import * as tf from '@tensorflow/tfjs';
import path from 'path';
import fs from 'fs';

import { loadOBD2Records } from './dataLoader.js';
import { buildRowFromRecord, isRowUsable, normalizeRow } from './dataPipeline.js';
import { computeReconstructionErrors, reconstructionErrorToHealthScore } from './model.js';
import { fileSystemLoadIO } from './ioHandler.js';
import { SYSTEM_IDS } from './systems.js';

const DEFAULT_CSV_PATH = path.join(process.cwd(), '..', 'Downloads', 'autoshop.obd2datapoints.csv');
const MODELS_DIR = path.join(process.cwd(), 'models');
const HEALTH_MODEL_BASE = path.join(MODELS_DIR, 'obd2-health-model');

async function main() {
  const csvPath = process.argv[2] || DEFAULT_CSV_PATH;

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
    if (!systems[systemId]) continue;
    const modelPath = path.join(HEALTH_MODEL_BASE, systemId);
    if (!fs.existsSync(path.join(modelPath, 'model.json'))) continue;
    models[systemId] = await tf.loadLayersModel(fileSystemLoadIO(modelPath));
  }

  if (Object.keys(models).length === 0) {
    console.error('No system models found. Run: npm run train');
    process.exit(1);
  }

  console.log('Loading data from:', csvPath);
  const { records } = loadOBD2Records(csvPath);
  const minRowCompleteness = Number(process.env.PREDICT_MIN_ROW_COMPLETENESS || 0.5);

  const allScores = [];

  for (let sampleIdx = 0; sampleIdx < records.length; sampleIdx++) {
    const record = records[sampleIdx];
    const scores = {};

    for (const systemId of SYSTEM_IDS) {
      if (!models[systemId] || !systems[systemId]) continue;
      const systemConfig = systems[systemId];
      const stats = systemConfig.stats;
      const systemFeatureNames = systemConfig.featureNames || [];

      if (!stats?.mins || !stats?.maxs || systemFeatureNames.length === 0) {
        continue;
      }

      const cleanedRow = buildRowFromRecord(record, systemFeatureNames);
      if (!isRowUsable(cleanedRow, minRowCompleteness)) continue;

      const normalizedRow = normalizeRow(cleanedRow, stats);
      const inputTensor = tf.tensor2d([normalizedRow]);

      const errors = computeReconstructionErrors(models[systemId], inputTensor);
      const errorVal = errors.dataSync()[0];

      scores[systemId] = reconstructionErrorToHealthScore(errorVal, systemConfig.threshold);

      tf.dispose([inputTensor, errors]);
    }

    if (Object.keys(scores).length > 0) {
      allScores.push({ index: sampleIdx, scores });
    }
  }

  console.log('\n--- Health Scores ---');
  console.log(`Total samples scored: ${allScores.length}/${records.length}`);
  console.log('');

  const limit = Math.min(20, allScores.length);
  for (let i = 0; i < limit; i++) {
    const { index, scores } = allScores[i];
    const parts = Object.entries(scores)
      .map(([k, v]) => `${k}=${v.toFixed(1)}`)
      .join(', ');
    console.log(`  Sample ${index}: ${parts}`);
  }

  if (allScores.length > limit) {
    console.log(`  ... and ${allScores.length - limit} more`);
  }

  // Summary: mean health per system
  const meanBySystem = {};
  for (const systemId of SYSTEM_IDS) {
    const vals = allScores.map(s => s.scores[systemId]).filter(v => v !== undefined);
    if (vals.length > 0) {
      meanBySystem[systemId] = vals.reduce((a, b) => a + b, 0) / vals.length;
    }
  }
  console.log('\nMean health by system:');
  for (const [k, v] of Object.entries(meanBySystem)) {
    console.log(`  ${k}: ${v.toFixed(1)}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
