import {
  createEmptyReviewDraft,
  getReviewProgress,
  type ReviewDraft,
} from './reviewDraft.ts'
import type { ReviewDataset, ReviewItem } from './reviewDataset.ts'
import type { ReviewRecord, ReviewSubmission } from './reviewRecord.ts'

export const UI_VERSION = '2'

export function createReviewSubmission(
  dataset: ReviewDataset,
  item: ReviewItem,
  draft: ReviewDraft,
): ReviewSubmission {
  const progress = getReviewProgress(dataset.form, item.candidates, draft)
  if (!progress.completed || draft.reviewStatus === null) {
    throw new Error('すべての設問へ回答してから保存してください。')
  }

  return {
    schemaVersion: 2,
    datasetId: dataset.datasetId,
    datasetVersion: dataset.datasetVersion,
    itemId: item.id,
    status: 'completed',
    candidateAnswers: item.candidates
      .filter((candidate) => candidate.status === 'available')
      .map((candidate) => ({
        candidateId: candidate.id,
        segment: { ...candidate.segment },
        answers: { ...draft.candidateAnswers[candidate.id] },
      })),
    reviewStatus: draft.reviewStatus,
    skipReason: null,
    reviewerKind: 'non_expert',
    protocol: { ...dataset.protocol },
    uiVersion: UI_VERSION,
  }
}

export function reviewRecordToDraft(record: ReviewRecord): ReviewDraft {
  const draft = createEmptyReviewDraft()
  record.candidateAnswers.forEach((candidate) => {
    draft.candidateAnswers[candidate.candidateId] = { ...candidate.answers }
  })
  draft.reviewStatus = record.reviewStatus
  return draft
}
