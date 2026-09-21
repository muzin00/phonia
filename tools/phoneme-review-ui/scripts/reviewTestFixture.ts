import type {
  ReviewRecord,
  ReviewSubmission,
} from '../src/domain/reviewRecord.ts'

export function reviewSubmissionFixture(): ReviewSubmission {
  return {
    schemaVersion: 1,
    datasetId: 'dataset-1',
    datasetVersion: '1',
    itemId: 'item-1',
    status: 'completed',
    candidateAnswers: [
      {
        candidateId: 'A',
        segment: { startSec: 1, endSec: 2 },
        answers: { audible: 'yes' },
      },
    ],
    comparison: { outcome: 'candidate', candidateId: 'A' },
    skipReason: null,
    reviewerKind: 'non_expert',
    protocol: { id: 'protocol-1', version: '1' },
    uiVersion: '1',
  }
}

export function reviewRecordFixture(revision = 1): ReviewRecord {
  return {
    ...reviewSubmissionFixture(),
    revision,
    recordedAt: `2026-09-22T00:00:0${revision}.000Z`,
  }
}

