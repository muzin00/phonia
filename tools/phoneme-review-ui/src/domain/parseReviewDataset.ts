import {
  REVIEW_DATASET_SCHEMA_VERSION,
  type ReviewDataset,
} from './reviewDataset.ts'

type JsonObject = Record<string, unknown>

export class InvalidReviewDatasetError extends Error {
  constructor(path: string, message: string) {
    super(`${path}: ${message}`)
    this.name = 'InvalidReviewDatasetError'
  }
}

export function parseReviewDataset(value: unknown): ReviewDataset {
  const dataset = objectAt(value, '$')

  equalAt(
    dataset.schemaVersion,
    REVIEW_DATASET_SCHEMA_VERSION,
    '$.schemaVersion',
  )
  nonEmptyStringAt(dataset.datasetId, '$.datasetId')
  nonEmptyStringAt(dataset.datasetVersion, '$.datasetVersion')
  nonEmptyStringAt(dataset.title, '$.title')
  validateProtocol(dataset.protocol, '$.protocol')
  validatePlayback(dataset.playback, '$.playback')
  validateForm(dataset.form, '$.form')

  const items = arrayAt(dataset.items, '$.items')
  if (items.length === 0) {
    fail('$.items', 'must contain at least one item')
  }

  const itemIds = new Set<string>()
  items.forEach((item, index) => {
    const path = `$.items[${index}]`
    const itemObject = objectAt(item, path)
    const itemId = nonEmptyStringAt(itemObject.id, `${path}.id`)
    uniqueAt(itemIds, itemId, `${path}.id`)
    validateUtterance(itemObject.utterance, `${path}.utterance`)
    validateTarget(itemObject.target, `${path}.target`)
    validateCandidates(itemObject.candidates, `${path}.candidates`)
  })

  return value as ReviewDataset
}

function validateProtocol(value: unknown, path: string) {
  const protocol = objectAt(value, path)
  nonEmptyStringAt(protocol.id, `${path}.id`)
  nonEmptyStringAt(protocol.version, `${path}.version`)
}

function validatePlayback(value: unknown, path: string) {
  const playback = objectAt(value, path)
  nonNegativeNumberAt(playback.contextPaddingSec, `${path}.contextPaddingSec`)
  positiveNumberAt(playback.boundaryLoopSec, `${path}.boundaryLoopSec`)
}

function validateForm(value: unknown, path: string) {
  const form = objectAt(value, path)
  const questions = arrayAt(
    form.candidateQuestions,
    `${path}.candidateQuestions`,
  )
  if (questions.length === 0) {
    fail(`${path}.candidateQuestions`, 'must contain at least one question')
  }

  const questionIds = new Set<string>()
  questions.forEach((question, questionIndex) => {
    const questionPath = `${path}.candidateQuestions[${questionIndex}]`
    const questionObject = objectAt(question, questionPath)
    const questionId = nonEmptyStringAt(
      questionObject.id,
      `${questionPath}.id`,
    )
    uniqueAt(questionIds, questionId, `${questionPath}.id`)
    nonEmptyStringAt(questionObject.prompt, `${questionPath}.prompt`)

    const choices = arrayAt(questionObject.choices, `${questionPath}.choices`)
    if (choices.length < 2) {
      fail(`${questionPath}.choices`, 'must contain at least two choices')
    }

    const choiceValues = new Set<string>()
    choices.forEach((choice, choiceIndex) => {
      const choicePath = `${questionPath}.choices[${choiceIndex}]`
      const choiceObject = objectAt(choice, choicePath)
      const choiceValue = nonEmptyStringAt(
        choiceObject.value,
        `${choicePath}.value`,
      )
      uniqueAt(choiceValues, choiceValue, `${choicePath}.value`)
      nonEmptyStringAt(choiceObject.label, `${choicePath}.label`)
    })
  })

  const comparison = objectAt(form.comparison, `${path}.comparison`)
  nonEmptyStringAt(comparison.prompt, `${path}.comparison.prompt`)
  booleanAt(
    comparison.allowIndistinguishable,
    `${path}.comparison.allowIndistinguishable`,
  )
  booleanAt(comparison.allowNone, `${path}.comparison.allowNone`)
}

function validateUtterance(value: unknown, path: string) {
  const utterance = objectAt(value, path)
  nonEmptyStringAt(utterance.id, `${path}.id`)
  nonEmptyStringAt(utterance.text, `${path}.text`)
  nonEmptyStringAt(utterance.audioUrl, `${path}.audioUrl`)
}

function validateTarget(value: unknown, path: string) {
  const target = objectAt(value, path)
  nonEmptyStringAt(target.label, `${path}.label`)
  nonNegativeIntegerAt(target.index, `${path}.index`)
  positiveIntegerAt(target.unitCount, `${path}.unitCount`)

  arrayAt(target.tags, `${path}.tags`).forEach((tag, index) => {
    nonEmptyStringAt(tag, `${path}.tags[${index}]`)
  })

  if (target.textRange !== undefined) {
    const textRange = objectAt(target.textRange, `${path}.textRange`)
    const start = nonNegativeIntegerAt(
      textRange.start,
      `${path}.textRange.start`,
    )
    const end = positiveIntegerAt(textRange.end, `${path}.textRange.end`)
    if (end <= start) {
      fail(`${path}.textRange.end`, 'must be greater than start')
    }
  }
}

function validateCandidates(value: unknown, path: string) {
  const candidates = arrayAt(value, path)
  if (candidates.length === 0) {
    fail(path, 'must contain at least one candidate')
  }

  const candidateIds = new Set<string>()
  let availableCount = 0

  candidates.forEach((candidate, index) => {
    const candidatePath = `${path}[${index}]`
    const candidateObject = objectAt(candidate, candidatePath)
    const candidateId = nonEmptyStringAt(
      candidateObject.id,
      `${candidatePath}.id`,
    )
    uniqueAt(candidateIds, candidateId, `${candidatePath}.id`)

    if (candidateObject.status === 'missing') {
      return
    }
    equalAt(candidateObject.status, 'available', `${candidatePath}.status`)
    availableCount += 1

    const segment = objectAt(
      candidateObject.segment,
      `${candidatePath}.segment`,
    )
    const startSec = nonNegativeNumberAt(
      segment.startSec,
      `${candidatePath}.segment.startSec`,
    )
    const endSec = positiveNumberAt(
      segment.endSec,
      `${candidatePath}.segment.endSec`,
    )
    if (endSec <= startSec) {
      fail(`${candidatePath}.segment.endSec`, 'must be greater than startSec')
    }
  })

  if (availableCount === 0) {
    fail(path, 'must contain at least one available candidate')
  }
}

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

function booleanAt(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    fail(path, 'must be a boolean')
  }
  return value
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

function nonNegativeIntegerAt(value: unknown, path: string): number {
  const number = nonNegativeNumberAt(value, path)
  if (!Number.isInteger(number)) {
    fail(path, 'must be an integer')
  }
  return number
}

function positiveIntegerAt(value: unknown, path: string): number {
  const number = nonNegativeIntegerAt(value, path)
  if (number === 0) {
    fail(path, 'must be greater than zero')
  }
  return number
}

function equalAt<T>(value: unknown, expected: T, path: string): asserts value is T {
  if (value !== expected) {
    fail(path, `must be ${JSON.stringify(expected)}`)
  }
}

function uniqueAt(values: Set<string>, value: string, path: string) {
  if (values.has(value)) {
    fail(path, `duplicate value ${JSON.stringify(value)}`)
  }
  values.add(value)
}

function fail(path: string, message: string): never {
  throw new InvalidReviewDatasetError(path, message)
}

