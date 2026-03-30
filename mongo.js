import { writeFile } from 'node:fs/promises';
import { MongoClient } from 'mongodb';

/*
 * Requires the MongoDB Node.js Driver
 * https://mongodb.github.io/node-mongodb-native
 */

const filter = {
  'timestamp': {
    '$gte': new Date('Sun, 15 Mar 2026 00:00:00 GMT'), 
    '$lt': new Date('Mon, 16 Mar 2026 00:00:00 GMT')
  }
};

const outputPath = 'obd2datapoints.csv';

function flattenDocument(value, prefix = '', output = {}) {
  if (value === null || value === undefined) {
    if (prefix) output[prefix] = '';
    return output;
  }

  if (value instanceof Date) {
    output[prefix] = value.toISOString();
    return output;
  }

  if (Array.isArray(value)) {
    output[prefix] = JSON.stringify(value);
    return output;
  }

  if (typeof value === 'object') {
    for (const [key, nestedValue] of Object.entries(value)) {
      const nextPrefix = prefix ? `${prefix}.${key}` : key;
      flattenDocument(nestedValue, nextPrefix, output);
    }
    return output;
  }

  output[prefix] = value;
  return output;
}

function escapeCsv(value) {
  const stringValue = value === null || value === undefined ? '' : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
}

const client = await MongoClient.connect(
  'mongodb://192.168.1.125:27018/automotiveai'
);

try {
  const coll = client.db('automotiveai').collection('obd2datapoints');
  const cursor = coll.find(filter);
  const result = await cursor.toArray();
  const flattenedRows = result.map((document) => flattenDocument(document));
  const headers = [...new Set(flattenedRows.flatMap((row) => Object.keys(row)))];
  const csvRows = [
    headers.join(','),
    ...flattenedRows.map((row) => headers.map((header) => escapeCsv(row[header])).join(','))
  ];

  await writeFile(outputPath, csvRows.join('\n'), 'utf8');
  console.log(`Exported ${result.length} rows to ${outputPath}`);
} finally {
  await client.close();
}