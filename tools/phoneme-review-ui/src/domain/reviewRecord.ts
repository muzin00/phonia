import type { ReviewStatus } from './reviewDataset.ts'

export type CandidateAnswerRecord = {
  candidateId: string
  segment: {
    startSec: number
    endSec: number
  } | null
  answers: Record<string, string>
}

export type ReviewSubmission = {
  schemaVersion: 2
  datasetId: string
  datasetVersion: string
  itemId: string
  status: 'completed' | 'skipped'
  candidateAnswers: CandidateAnswerRecord[]
  reviewStatus: ReviewStatus | null
  skipReason: string | null
  reviewerKind: string
  protocol: {
    id: string
    version: string
  }
  uiVersion: string
}

export type ReviewRecord = ReviewSubmission & {
  revision: number
  recordedAt: string
}

export class InvalidReviewRecordError extends Error {
  constructor(path: string, message: string) {
    super(`${path}: ${message}`)
    this.name = 'InvalidReviewRecordError'
  }
}

export function parseReviewSubmission(value: unknown): ReviewSubmission {
  const submission = validateSubmission(value, '$')
  return submission as ReviewSubmission
}

export function parseReviewRecord(value: unknown): ReviewRecord {
  const record = validateSubmission(value, '$')
  positiveIntegerAt(record.revision, '$.revision')
  const recordedAt = nonEmptyStringAt(record.recordedAt, '$.recordedAt')
  if (Number.isNaN(Date.parse(recordedAt))) {
    fail('$.recordedAt', 'must be an ISO-compatible date-time')
  }
  return value as ReviewRecord
}

function validateSubmission(value: unknown, path: string): JsonObject {
  const submission = objectAt(value, path)
  equalAt(submission.schemaVersion, 2, `${path}.schemaVersion`)
  nonEmptyStringAt(submission.datasetId, `${path}.datasetId`)
  nonEmptyStringAt(submission.datasetVersion, `${path}.datasetVersion`)
  nonEmptyStringAt(submission.itemId, `${path}.itemId`)

  if (submission.status !== 'completed' && submission.status !== 'skipped') {
    fail(`${path}.status`, 'must be "completed" or "skipped"')
  }

  const candidateAnswers = arrayAt(
    submission.candidateAnswers,
    `${path}.candidateAnswers`,
  )
  const candidateIds = new Set<string>()
  candidateAnswers.forEach((answer, index) => {
    const answerPath = `${path}.candidateAnswers[${index}]`
    const answerObject = objectAt(answer, answerPath)
    const candidateId = nonEmptyStringAt(
      answerObject.candidateId,
      `${answerPath}.candidateId`,
    )
    if (candidateIds.has(candidateId)) {
      fail(`${answerPath}.candidateId`, `duplicate value ${JSON.stringify(candidateId)}`)
    }
    candidateIds.add(candidateId)

    if (answerObject.segment !== null) {
      const segment = objectAt(answerObject.segment, `${answerPath}.segment`)
      const startSec = nonNegativeNumberAt(
        segment.startSec,
        `${answerPath}.segment.startSec`,
      )
      const endSec = positiveNumberAt(
        segment.endSec,
        `${answerPath}.segment.endSec`,
      )
      if (endSec <= startSec) {
        fail(`${answerPath}.segment.endSec`, 'must be greater than startSec')
      }
    }

    const answers = objectAt(answerObject.answers, `${answerPath}.answers`)
    Object.entries(answers).forEach(([questionId, answerValue]) => {
      nonEmptyStringAt(questionId, `${answerPath}.answers key`)
      nonEmptyStringAt(answerValue, `${answerPath}.answers.${questionId}`)
    })
  })

  validateReviewStatus(submission.reviewStatus, `${path}.reviewStatus`)
  nullableStringAt(submission.skipReason, `${path}.skipReason`)
  nonEmptyStringAt(submission.reviewerKind, `${path}.reviewerKind`)

  const protocol = objectAt(submission.protocol, `${path}.protocol`)
  nonEmptyStringAt(protocol.id, `${path}.protocol.id`)
  nonEmptyStringAt(protocol.version, `${path}.protocol.version`)
  nonEmptyStringAt(submission.uiVersion, `${path}.uiVersion`)

  if (submission.status === 'completed') {
    if (candidateAnswers.length === 0) {
      fail(`${path}.candidateAnswers`, 'must not be empty for a completed review')
    }
    if (submission.reviewStatus === null) {
      fail(`${path}.reviewStatus`, 'must not be null for a completed review')
    }
    if (submission.skipReason !== null) {
      fail(`${path}.skipReason`, 'must be null for a completed review')
    }
  } else if (
    typeof submission.skipReason !== 'string' ||
    submission.skipReason.trim().length === 0
  ) {
    fail(`${path}.skipReason`, 'must explain why the review was skipped')
  }

  return submission
}

function validateReviewStatus(value: unknown, path: string) {
  if (value === null) {
    return
  }
  if (value !== 'accepted' && value !== 'rejected' && value !== 'uncertain') {
    fail(path, 'has an unsupported value')
  }
}

type JsonObject = Record<string, unknown>

function objectAt(value: unknown, path: string): JsonObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    fail(path, 'must be an object')
  }
  return value as JsonObject
}

function arrayAt(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    fail(path, 'must be an array')
  }
  return value
}

function nonEmptyStringAt(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.trim().length === 0) {
    fail(path, 'must be a non-empty string')
  }
  return value
}

function nullableStringAt(value: unknown, path: string): string | null {
  if (value === null) {
    return null
  }
  return nonEmptyStringAt(value, path)
}

function nonNegativeNumberAt(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    fail(path, 'must be a finite number greater than or equal to zero')
  }
  return value
}

function positiveNumberAt(value: unknown, path: string): number {
  const number = nonNegativeNumberAt(value, path)
  if (number === 0) {
    fail(path, 'must be greater than zero')
  }
  return number
}

function positiveIntegerAt(value: unknown, path: string): number {
  const number = positiveNumberAt(value, path)
  if (!Number.isInteger(number)) {
    fail(path, 'must be an integer')
  }
  return number
}

function equalAt<T>(value: unknown, expected: T, path: string): asserts value is T {
  if (value !== expected) {
    fail(path, `must be ${JSON.stringify(expected)}`)
  }
}

function fail(path: string, message: string): never {
  throw new InvalidReviewRecordError(path, message)
}
