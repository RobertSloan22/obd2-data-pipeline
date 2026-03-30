/**
 * Node.js file system IO handler for TensorFlow.js model save/load
 * Used when @tensorflow/tfjs-node is not available
 */

import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const mkdir = promisify(fs.mkdir);

const MODEL_JSON = 'model.json';
const WEIGHTS_BIN = 'weights.bin';

/**
 * Create IO handler for saving to directory
 * @param {string} dirPath - Absolute path to directory
 * @returns {object} IOHandler for model.save()
 */
export function fileSystemIO(dirPath) {
  const resolved = path.resolve(dirPath);

  return {
    async save(modelArtifacts) {
      await mkdir(resolved, { recursive: true });

      const weightsManifest = [{
        paths: [WEIGHTS_BIN],
        weights: modelArtifacts.weightSpecs
      }];

      const modelJSON = {
        modelTopology: modelArtifacts.modelTopology,
        weightsManifest,
        format: modelArtifacts.format,
        generatedBy: modelArtifacts.generatedBy,
        convertedBy: modelArtifacts.convertedBy
      };
      if (modelArtifacts.trainingConfig) modelJSON.trainingConfig = modelArtifacts.trainingConfig;
      if (modelArtifacts.signature) modelJSON.signature = modelArtifacts.signature;
      if (modelArtifacts.userDefinedMetadata) modelJSON.userDefinedMetadata = modelArtifacts.userDefinedMetadata;

      await writeFile(path.join(resolved, MODEL_JSON), JSON.stringify(modelJSON), 'utf8');
      await writeFile(path.join(resolved, WEIGHTS_BIN), Buffer.from(modelArtifacts.weightData), 'binary');

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          modelTopologyBytes: JSON.stringify(modelArtifacts.modelTopology).length,
          weightSpecs: modelArtifacts.weightSpecs,
          weightDataBytes: modelArtifacts.weightData.byteLength
        }
      };
    }
  };
}

/**
 * Create IO handler for loading model from directory
 * @param {string} modelPath - Path to directory containing model.json
 * @returns {object} IOHandler for tf.loadLayersModel()
 */
export function fileSystemLoadIO(modelPath) {
  const resolved = path.resolve(modelPath);

  return {
    async load() {
      const jsonPath = path.join(resolved, MODEL_JSON);
      const modelJSON = JSON.parse(await readFile(jsonPath, 'utf8'));
      const dir = path.dirname(jsonPath);

      const weightsManifest = modelJSON.weightsManifest || modelJSON.weights;
      let weightData;
      let weightSpecs = [];

      if (weightsManifest && weightsManifest.length > 0) {
        const buffers = [];
        for (const group of weightsManifest) {
          for (const p of group.paths) {
            const buf = await readFile(path.join(dir, p));
            buffers.push(buf);
          }
          weightSpecs = weightSpecs.concat(group.weights || []);
        }
        const totalLen = buffers.reduce((s, b) => s + b.length, 0);
        weightData = new ArrayBuffer(totalLen);
        const view = new Uint8Array(weightData);
        let offset = 0;
        for (const b of buffers) {
          view.set(new Uint8Array(b), offset);
          offset += b.length;
        }
      }

      return {
        modelTopology: modelJSON.modelTopology,
        weightSpecs,
        weightData: weightData || new ArrayBuffer(0)
      };
    }
  };
}
