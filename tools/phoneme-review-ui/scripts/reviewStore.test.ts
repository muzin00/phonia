import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'

import {
  appendReviewSubmission,
  readReviewRecords,
  selectLatestReviewRecords,
} from '../vite/reviewStore.ts'
import { reviewSubmissionFixture } from './reviewTestFixture.ts'

test('appends immutable revisions and selects the latest record', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'phonia-review-store-'))
  const filePath = join(directory, 'review-records.jsonl')

  try {
    const first = await appendReviewSubmission(
      filePath,
      reviewSubmissionFixture(),
      '2026-09-22T00:00:01.000Z',
    )
    const second = await appendReviewSubmission(
      filePath,
      reviewSubmissionFixture(),
      '2026-09-22T00:00:02.000Z',
    )

    assert.equal(first.revision, 1)
    assert.equal(second.revision, 2)

    const records = await readReviewRecords(filePath)
    assert.equal(records.length, 2)
    assert.deepEqual(
      selectLatestReviewRecords(records, 'dataset-1', '1'),
      [second],
    )
    assert.equal((await readFile(filePath, 'utf8')).trim().split('\n').length, 2)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('returns no records when the output file does not exist', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'phonia-review-store-'))
  try {
    assert.deepEqual(await readReviewRecords(join(directory, 'missing.jsonl')), [])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

