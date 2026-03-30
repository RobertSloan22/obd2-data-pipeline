/**
 * Autoencoder model for OBD2 anomaly detection
 * Trained on normal data; high reconstruction error indicates anomaly
 */

import * as tf from '@tensorflow/tfjs';

/**
 * Build autoencoder model for anomaly detection
 * @param {number} inputDim - Number of input features
 * @param {object} options - Model options
 * @returns {tf.Sequential}
 */
export function buildAutoencoder(inputDim, options = {}) {
  const {
    encodingDims = [64, 32, 16],
    activation = 'relu',
    latentActivation = 'relu'
  } = options;

  const model = tf.sequential();

  // Encoder
  let prevDim = inputDim;
  for (const dim of encodingDims) {
    model.add(tf.layers.dense({
      units: dim,
      activation,
      inputShape: prevDim === inputDim ? [inputDim] : undefined
    }));
    prevDim = dim;
  }

  // Latent / bottleneck
  const latentDim = Math.max(8, Math.floor(encodingDims[encodingDims.length - 1] / 2));
  model.add(tf.layers.dense({
    units: latentDim,
    activation: latentActivation,
    name: 'latent'
  }));

  // Decoder (mirror encoder)
  const decoderDims = [...encodingDims].reverse();
  for (const dim of decoderDims) {
    model.add(tf.layers.dense({
      units: dim,
      activation
    }));
  }

  model.add(tf.layers.dense({
    units: inputDim,
    activation: 'sigmoid',
    name: 'output'
  }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'meanSquaredError',
    metrics: ['mse']
  });

  return model;
}

/**
 * Train the autoencoder on normalized data
 * @param {tf.Sequential} model - Autoencoder model
 * @param {tf.Tensor2D} trainData - Normalized training data
 * @param {object} options - Training options
 * @returns {object} - Training history
 */
export async function trainModel(model, trainData, options = {}) {
  const {
    epochs = 50,
    batchSize = 32,
    validationSplit = 0.1,
    verbose = 1
  } = options;

  const history = await model.fit(trainData, trainData, {
    epochs,
    batchSize: Math.min(batchSize, Math.floor(trainData.shape[0] / 4)),
    validationSplit,
    shuffle: true,
    verbose: verbose ? 1 : 0
  });

  return history;
}

/**
 * Train the autoencoder from normalized batches produced on demand.
 * Suitable for large datasets that should not be loaded entirely in memory.
 * @param {tf.Sequential} model - Autoencoder model
 * @param {Function} createBatchIterator - Function returning an async iterator of number[][]
 * @param {object} options - Training options
 * @returns {object} - Epoch loss history
 */
export async function trainModelOnBatches(model, createBatchIterator, options = {}) {
  const {
    epochs = 50,
    verbose = 1
  } = options;

  const history = {
    loss: []
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    let totalLoss = 0;
    let totalSamples = 0;
    let batchCount = 0;

    for await (const batch of createBatchIterator()) {
      if (!batch || batch.length === 0) {
        continue;
      }

      const batchTensor = tf.tensor2d(batch);
      const fitHistory = await model.fit(batchTensor, batchTensor, {
        epochs: 1,
        batchSize: batch.length,
        shuffle: false,
        verbose: 0
      });

      const batchLoss = fitHistory.history.loss[fitHistory.history.loss.length - 1] ?? 0;
      totalLoss += batchLoss * batch.length;
      totalSamples += batch.length;
      batchCount += 1;

      tf.dispose(batchTensor);
    }

    if (totalSamples === 0) {
      throw new Error('No batches were produced for training.');
    }

    const meanLoss = totalLoss / totalSamples;
    history.loss.push(meanLoss);

    if (verbose) {
      console.log(
        `Epoch ${epoch + 1}/${epochs} - loss=${meanLoss.toFixed(6)} batches=${batchCount}`
      );
    }
  }

  return history;
}

/**
 * Compute reconstruction errors for each sample
 * @param {tf.Sequential} model - Trained autoencoder
 * @param {tf.Tensor2D} data - Input data tensor
 * @returns {tf.Tensor1D} - Reconstruction error (MSE) per sample
 */
export function computeReconstructionErrors(model, data) {
  const reconstructed = model.predict(data);
  const diff = tf.sub(data, reconstructed);
  const sqDiff = tf.square(diff);
  const mse = tf.mean(sqDiff, 1); // Mean over features, keep batch dim
  return mse;
}

/**
 * Map reconstruction error to health score 0-100
 * health = max(0, min(100, 100 * (1 - error / threshold)))
 * @param {number} error - Reconstruction error (MSE)
 * @param {number} threshold - P99 reconstruction error from training
 * @returns {number} Health score in [0, 100]
 */
export function reconstructionErrorToHealthScore(error, threshold) {
  if (threshold <= 0) return 100;
  const health = 100 * (1 - error / threshold);
  return Math.max(0, Math.min(100, health));
}
