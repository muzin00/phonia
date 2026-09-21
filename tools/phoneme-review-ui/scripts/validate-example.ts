import { readFile } from 'node:fs/promises'

import { parseReviewDataset } from '../src/domain/parseReviewDataset.ts'

const exampleUrl = new URL('../examples/review-dataset.json', import.meta.url)
const source = await readFile(exampleUrl, 'utf8')
const dataset = parseReviewDataset(JSON.parse(source) as unknown)

console.log(
  `Valid review dataset: ${dataset.datasetId}@${dataset.datasetVersion} (${dataset.items.length} item)`,
)

