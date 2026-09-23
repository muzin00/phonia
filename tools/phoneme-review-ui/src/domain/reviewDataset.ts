export const REVIEW_DATASET_SCHEMA_VERSION = 2 as const

export type ReviewStatus = 'accepted' | 'rejected' | 'uncertain'

export type ReviewDataset = {
  schemaVersion: typeof REVIEW_DATASET_SCHEMA_VERSION
  datasetId: string
  datasetVersion: string
  title: string
  protocol: {
    id: string
    version: string
  }
  playback: {
    contextPaddingSec: number
  }
  form: ReviewForm
  items: ReviewItem[]
}

export type ReviewForm = {
  candidateQuestions: CandidateQuestion[]
  reviewStatus: {
    prompt: string
    choices: Array<{
      value: ReviewStatus
      label: string
    }>
  }
}

export type CandidateQuestion = {
  id: string
  prompt: string
  choices: Array<{
    value: string
    label: string
  }>
}

export type ReviewItem = {
  id: string
  utterance: {
    id: string
    text: string
    audioUrl: string
  }
  target: {
    label: string
    index: number
    unitCount: number
    tags: string[]
    textRange?: {
      start: number
      end: number
    }
  }
  candidates: ReviewCandidate[]
}

export type ReviewCandidate =
  | {
      id: string
      status: 'available'
      segment: {
        startSec: number
        endSec: number
      }
    }
  | {
      id: string
      status: 'missing'
    }
