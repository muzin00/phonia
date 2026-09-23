import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import { parseReviewDataset } from '../src/domain/parseReviewDataset.ts'
import {
  answerCandidateQuestion,
  answerReviewStatus,
  createEmptyReviewDraft,
} from '../src/domain/reviewDraft.ts'
import {
  createReviewSubmission,
  reviewRecordToDraft,
} from '../src/domain/reviewPersistence.ts'
import {
  InvalidReviewRecordError,
  parseReviewRecord,
  parseReviewSubmission,
} from '../src/domain/reviewRecord.ts'
import { reviewRecordFixture, reviewSubmissionFixture } from './reviewTestFixture.ts'

test('accepts a completed review submission and record', () => {
  assert.deepEqual(
    parseReviewSubmission(reviewSubmissionFixture()),
    reviewSubmissionFixture(),
  )
  assert.deepEqual(parseReviewRecord(reviewRecordFixture()), reviewRecordFixture())
})

test('rejects a completed submission without a review status', () => {
  const invalid = { ...reviewSubmissionFixture(), reviewStatus: null }
  assert.throws(
    () => parseReviewSubmission(invalid),
    (error: unknown) => {
      assert.ok(error instanceof InvalidReviewRecordError)
      assert.match(error.message, /reviewStatus/)
      return true
    },
  )
})

test('converts a completed draft to a submission and restores it', async () => {
  const exampleUrl = new URL(
    '../public/examples/review-dataset.json',
    import.meta.url,
  )
  const dataset = parseReviewDataset(
    JSON.parse(await readFile(exampleUrl, 'utf8')) as unknown,
  )
  const item = dataset.items[0]!
  let draft = createEmptyReviewDraft()
  dataset.form.candidateQuestions.forEach((question) => {
    draft = answerCandidateQuestion(
      draft,
      'A',
      question.id,
      question.choices[0]!.value,
    )
  })
  draft = answerReviewStatus(draft, 'accepted')

  const submission = createReviewSubmission(dataset, item, draft)
  const restored = reviewRecordToDraft({
    ...submission,
    revision: 1,
    recordedAt: '2026-09-22T00:00:00.000Z',
  })

  assert.deepEqual(restored, draft)
  assert.deepEqual(submission.candidateAnswers[0]!.segment, {
    startSec: 1.86,
    endSec: 1.95,
  })
})
