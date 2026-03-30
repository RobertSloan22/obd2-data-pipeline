/**
 * Shared data-cleaning utilities used by both training and prediction.
 * Keeps numeric coercion, completeness filtering, and normalization consistent.
 */

export function toFiniteNumber(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (trimmed === '') {
      return null;
    }

    const parsed = Number(trimmed);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

export function buildRowFromRecord(record, featureNames) {
  return featureNames.map((featureName) => toFiniteNumber(record?.[featureName]));
}

export function getRowPresence(row) {
  return row.reduce((count, value) => count + (value !== null ? 1 : 0), 0);
}

export function isRowUsable(row, minRowCompleteness = 0.8) {
  if (!Array.isArray(row) || row.length === 0) {
    return false;
  }

  return getRowPresence(row) / row.length >= minRowCompleteness;
}

export function normalizeRow(row, stats) {
  return row.map((value, index) => {
    const min = stats.mins[index];
    const max = stats.maxs[index];
    const fallback = stats.means?.[index] ?? min;
    const rawValue = value === null ? fallback : value;
    return (rawValue - min) / (max - min);
  });
}
