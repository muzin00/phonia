import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

import {
  InvalidReviewDatasetError,
  parseReviewDataset,
} from '../src/domain/parseReviewDataset.ts'

const exampleUrl = new URL(
  '../public/examples/review-dataset.json',
  import.meta.url,
)
const example = JSON.parse(await readFile(exampleUrl, 'utf8')) as unknown

test('accepts the documented example dataset', () => {
  const dataset = parseReviewDataset(example)

  assert.equal(dataset.datasetId, 'mfa-vowel-pilot')
  assert.equal(dataset.datasetVersion, '5')
  assert.equal(dataset.protocol.version, '5')
  assert.deepEqual(
    dataset.form.candidateQuestions.map((question) => question.id),
    ['perceived_content'],
  )
  assert.deepEqual(
    dataset.form.candidateQuestions[0]?.choices.map((choice) => choice.value),
    [
      'target_vowel',
      'target_vowel_with_non_vowel',
      'other_vowel',
      'non_vowel_only',
      'near_silence',
      'uncertain',
    ],
  )
  assert.equal(dataset.items.length, 1)
})

test('rejects an available segment whose end is before its start', () => {
  const invalid = structuredClone(example) as {
    items: Array<{
      candidates: Array<{ segment?: { endSec: number } }>
    }>
  }
  invalid.items[0]!.candidates[0]!.segment!.endSec = 1

  assert.throws(
    () => parseReviewDataset(invalid),
    (error: unknown) => {
      assert.ok(error instanceof InvalidReviewDatasetError)
      assert.match(error.message, /segment\.endSec: must be greater than startSec/)
      return true
    },
  )
})

test('rejects duplicate candidate IDs within an item', () => {
  const invalid = structuredClone(example) as {
    items: Array<{
      candidates: Array<{ id: string }>
    }>
  }
  invalid.items[0]!.candidates.push({ id: 'A' })

  assert.throws(
    () => parseReviewDataset(invalid),
    (error: unknown) => {
      assert.ok(error instanceof InvalidReviewDatasetError)
      assert.match(error.message, /duplicate value "A"/)
      return true
    },
  )
})

test('rejects multiple available candidates in the single-candidate protocol', () => {
  const invalid = structuredClone(example) as {
    items: Array<{
      candidates: Array<{
        id: string
        status: string
        segment?: { startSec: number; endSec: number }
      }>
    }>
  }
  invalid.items[0]!.candidates.push({
    id: 'B',
    status: 'available',
    segment: { startSec: 1.8, endSec: 1.96 },
  })

  assert.throws(
    () => parseReviewDataset(invalid),
    (error: unknown) => {
      assert.ok(error instanceof InvalidReviewDatasetError)
      assert.match(error.message, /exactly one available candidate/)
      return true
    },
  )
})
