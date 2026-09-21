import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

import { parseReviewDataset } from '../src/domain/parseReviewDataset.ts'

const inputPath = process.argv.slice(2).find((argument) => argument !== '--')
const datasetUrl = inputPath
  ? pathToFileURL(resolve(inputPath))
  : new URL('../public/examples/review-dataset.json', import.meta.url)
const source = await readFile(datasetUrl, 'utf8')
const dataset = parseReviewDataset(JSON.parse(source) as unknown)

console.log(
  `Valid review dataset: ${dataset.datasetId}@${dataset.datasetVersion} (${dataset.items.length} item)`,
)
