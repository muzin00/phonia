import type { ReviewDraft } from './reviewDraft.ts'

const LOCAL_REVIEW_SCHEMA_VERSION = 1
const STORAGE_PREFIX = 'phonia:review-drafts'

export type LocalReviewState = {
  currentIndex: number
  drafts: Record<string, ReviewDraft>
}

type StorageReader = Pick<Storage, 'getItem'>
type StorageWriter = Pick<Storage, 'setItem'>

export function localReviewStorageKey(
  datasetId: string,
  datasetVersion: string,
): string {
  return `${STORAGE_PREFIX}:${datasetId}:${datasetVersion}`
}

export function loadLocalReviewState(
  storage: StorageReader,
  datasetId: string,
  datasetVersion: string,
): LocalReviewState | null {
  const source = storage.getItem(
    localReviewStorageKey(datasetId, datasetVersion),
  )
  if (source === null) {
    return null
  }

  try {
    const value = JSON.parse(source) as unknown
    if (!isObject(value) || value.schemaVersion !== LOCAL_REVIEW_SCHEMA_VERSION) {
      return null
    }
    if (
      value.datasetId !== datasetId ||
      value.datasetVersion !== datasetVersion ||
      !Number.isInteger(value.currentIndex) ||
      Number(value.currentIndex) < 0 ||
      !isObject(value.drafts)
    ) {
      return null
    }

    const drafts: Record<string, ReviewDraft> = {}
    for (const [itemId, draftValue] of Object.entries(value.drafts)) {
      const draft = parseDraft(draftValue)
      if (draft === null) {
        return null
      }
      drafts[itemId] = draft
    }
    return { currentIndex: Number(value.currentIndex), drafts }
  } catch {
    return null
  }
}

export function saveLocalReviewState(
  storage: StorageWriter,
  datasetId: string,
  datasetVersion: string,
  state: LocalReviewState,
): void {
  storage.setItem(
    localReviewStorageKey(datasetId, datasetVersion),
    JSON.stringify({
      schemaVersion: LOCAL_REVIEW_SCHEMA_VERSION,
      datasetId,
      datasetVersion,
      currentIndex: state.currentIndex,
      drafts: state.drafts,
    }),
  )
}

function parseDraft(value: unknown): ReviewDraft | null {
  if (!isObject(value) || !isObject(value.candidateAnswers)) {
    return null
  }
  if (
    value.reviewStatus !== null &&
    value.reviewStatus !== 'accepted' &&
    value.reviewStatus !== 'rejected' &&
    value.reviewStatus !== 'uncertain'
  ) {
    return null
  }

  const candidateAnswers: Record<string, Record<string, string>> = {}
  for (const [candidateId, answersValue] of Object.entries(
    value.candidateAnswers,
  )) {
    if (!isObject(answersValue)) {
      return null
    }
    const answers: Record<string, string> = {}
    for (const [questionId, answer] of Object.entries(answersValue)) {
      if (typeof answer !== 'string') {
        return null
      }
      answers[questionId] = answer
    }
    candidateAnswers[candidateId] = answers
  }
  return {
    candidateAnswers,
    reviewStatus: value.reviewStatus,
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
