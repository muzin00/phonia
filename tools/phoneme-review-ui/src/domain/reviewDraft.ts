import type { ReviewCandidate, ReviewForm } from './reviewDataset.ts'

export type ComparisonAnswer =
  | { outcome: 'candidate'; candidateId: string }
  | { outcome: 'indistinguishable' | 'none' }

export type ReviewDraft = {
  candidateAnswers: Record<string, Record<string, string>>
  comparison: ComparisonAnswer | null
}

export type ReviewProgress = {
  answered: number
  total: number
  remaining: number
  completed: boolean
}

export function createEmptyReviewDraft(): ReviewDraft {
  return {
    candidateAnswers: {},
    comparison: null,
  }
}

export function answerCandidateQuestion(
  draft: ReviewDraft,
  candidateId: string,
  questionId: string,
  value: string,
): ReviewDraft {
  return {
    ...draft,
    candidateAnswers: {
      ...draft.candidateAnswers,
      [candidateId]: {
        ...draft.candidateAnswers[candidateId],
        [questionId]: value,
      },
    },
  }
}

export function answerComparison(
  draft: ReviewDraft,
  comparison: ComparisonAnswer,
): ReviewDraft {
  return { ...draft, comparison }
}

export function getReviewProgress(
  form: ReviewForm,
  candidates: ReviewCandidate[],
  draft: ReviewDraft,
): ReviewProgress {
  const availableCandidates = candidates.filter(
    (candidate) => candidate.status === 'available',
  )
  const candidateAnswerCount = availableCandidates.reduce(
    (total, candidate) =>
      total +
      form.candidateQuestions.filter(
        (question) => draft.candidateAnswers[candidate.id]?.[question.id] !== undefined,
      ).length,
    0,
  )
  const total = availableCandidates.length * form.candidateQuestions.length + 1
  const answered = candidateAnswerCount + (draft.comparison === null ? 0 : 1)

  return {
    answered,
    total,
    remaining: total - answered,
    completed: answered === total,
  }
}

