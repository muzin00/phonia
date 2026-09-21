import assert from 'node:assert/strict'
import test from 'node:test'

import {
  answerCandidateQuestion,
  answerComparison,
  createEmptyReviewDraft,
  getReviewProgress,
} from '../src/domain/reviewDraft.ts'
import type { ReviewCandidate, ReviewForm } from '../src/domain/reviewDataset.ts'

const form: ReviewForm = {
  candidateQuestions: [
    {
      id: 'audible',
      prompt: '聞き取れますか？',
      choices: [
        { value: 'yes', label: 'はい' },
        { value: 'no', label: 'いいえ' },
      ],
    },
    {
      id: 'clipped',
      prompt: '切れていますか？',
      choices: [
        { value: 'yes', label: 'はい' },
        { value: 'no', label: 'いいえ' },
      ],
    },
  ],
  comparison: {
    prompt: '最も自然な候補は？',
    allowIndistinguishable: true,
    allowNone: true,
  },
}

const candidates: ReviewCandidate[] = [
  {
    id: 'A',
    status: 'available',
    segment: { startSec: 1, endSec: 2 },
  },
  { id: 'B', status: 'missing' },
]

test('counts only questions for available candidates plus comparison', () => {
  assert.deepEqual(getReviewProgress(form, candidates, createEmptyReviewDraft()), {
    answered: 0,
    total: 3,
    remaining: 3,
    completed: false,
  })
})

test('preserves prior answers while answering another question', () => {
  const first = answerCandidateQuestion(
    createEmptyReviewDraft(),
    'A',
    'audible',
    'yes',
  )
  const second = answerCandidateQuestion(first, 'A', 'clipped', 'no')

  assert.deepEqual(second.candidateAnswers.A, {
    audible: 'yes',
    clipped: 'no',
  })
})

test('marks the draft complete when all required answers exist', () => {
  let draft = createEmptyReviewDraft()
  draft = answerCandidateQuestion(draft, 'A', 'audible', 'yes')
  draft = answerCandidateQuestion(draft, 'A', 'clipped', 'no')
  draft = answerComparison(draft, { outcome: 'candidate', candidateId: 'A' })

  assert.deepEqual(getReviewProgress(form, candidates, draft), {
    answered: 3,
    total: 3,
    remaining: 0,
    completed: true,
  })
})

