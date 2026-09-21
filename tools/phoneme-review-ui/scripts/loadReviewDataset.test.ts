import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import {
  loadReviewDataset,
  ReviewDatasetLoadError,
} from '../src/data/loadReviewDataset.ts'

const exampleUrl = new URL(
  '../public/examples/review-dataset.json',
  import.meta.url,
)
const exampleSource = await readFile(exampleUrl, 'utf8')

test('loads and validates a review dataset', async () => {
  const dataset = await loadReviewDataset(
    '/dataset.json',
    undefined,
    async () => new Response(exampleSource),
  )

  assert.equal(dataset.datasetId, 'mfa-vowel-pilot')
})

test('reports an HTTP failure with the status code', async () => {
  await assert.rejects(
    loadReviewDataset(
      '/missing.json',
      undefined,
      async () => new Response(null, { status: 404 }),
    ),
    (error: unknown) => {
      assert.ok(error instanceof ReviewDatasetLoadError)
      assert.match(error.message, /HTTP 404/)
      return true
    },
  )
})

test('rejects JSON that does not satisfy the data contract', async () => {
  await assert.rejects(
    loadReviewDataset(
      '/invalid.json',
      undefined,
      async () => Response.json({ schemaVersion: 1 }),
    ),
    (error: unknown) => {
      assert.ok(error instanceof ReviewDatasetLoadError)
      assert.match(error.message, /形式が正しくありません/)
      return true
    },
  )
})

