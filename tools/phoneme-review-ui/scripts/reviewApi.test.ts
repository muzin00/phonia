import assert from 'node:assert/strict'
import test from 'node:test'

import {
  loadLatestReviewRecords,
  saveReviewSubmission,
} from '../src/data/reviewApi.ts'
import { reviewRecordFixture, reviewSubmissionFixture } from './reviewTestFixture.ts'

test('loads the latest review records from the API', async () => {
  const records = await loadLatestReviewRecords(
    'dataset-1',
    '1',
    undefined,
    async (input) => {
      assert.match(String(input), /datasetId=dataset-1/)
      return Response.json({ records: [reviewRecordFixture()] })
    },
  )

  assert.deepEqual(records, [reviewRecordFixture()])
})

test('posts a submission and parses the saved record', async () => {
  const submission = reviewSubmissionFixture()
  const record = await saveReviewSubmission(submission, async (_input, init) => {
    assert.equal(init?.method, 'POST')
    assert.deepEqual(JSON.parse(String(init?.body)) as unknown, submission)
    return Response.json({ record: reviewRecordFixture() }, { status: 201 })
  })

  assert.deepEqual(record, reviewRecordFixture())
})

