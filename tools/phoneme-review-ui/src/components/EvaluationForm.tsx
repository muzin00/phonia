import {
  answerCandidateQuestion,
  answerComparison,
  getReviewProgress,
  type ComparisonAnswer,
  type ReviewDraft,
} from '../domain/reviewDraft.ts'
import type {
  ReviewCandidate,
  ReviewForm,
} from '../domain/reviewDataset.ts'

type AvailableCandidate = Extract<ReviewCandidate, { status: 'available' }>

export function EvaluationForm({
  form,
  candidates,
  selectedCandidateId,
  draft,
  onSelectCandidate,
  onChange,
}: {
  form: ReviewForm
  candidates: AvailableCandidate[]
  selectedCandidateId: string
  draft: ReviewDraft
  onSelectCandidate: (candidateId: string) => void
  onChange: (draft: ReviewDraft) => void
}) {
  const selectedCandidate = candidates.find(
    (candidate) => candidate.id === selectedCandidateId,
  )
  const progress = getReviewProgress(form, candidates, draft)

  if (selectedCandidate === undefined) {
    return null
  }

  return (
    <section className="evaluation-section" aria-labelledby="evaluation-heading">
      <div className="section-heading">
        <div>
          <p className="section-label">Evaluation</p>
          <h2 id="evaluation-heading">区間を評価する</h2>
        </div>
        <p
          className={`answer-progress${progress.completed ? ' answer-progress-complete' : ''}`}
          aria-live="polite"
        >
          {progress.completed
            ? '回答完了'
            : `未回答 ${progress.remaining} / ${progress.total}`}
        </p>
      </div>

      {candidates.length > 1 && (
        <div className="candidate-tabs" aria-label="評価する候補">
          {candidates.map((candidate) => {
            const answerCount = form.candidateQuestions.filter(
              (question) =>
                draft.candidateAnswers[candidate.id]?.[question.id] !== undefined,
            ).length
            return (
              <button
                key={candidate.id}
                type="button"
                className="candidate-tab"
                aria-pressed={candidate.id === selectedCandidateId}
                onClick={() => onSelectCandidate(candidate.id)}
              >
                候補{candidate.id}
                <span>
                  {answerCount}/{form.candidateQuestions.length}
                </span>
              </button>
            )
          })}
        </div>
      )}

      <div className="candidate-questions">
        <div className="question-group-heading">
          <span className="candidate-id">{selectedCandidate.id}</span>
          <div>
            <h3>候補{selectedCandidate.id}の評価</h3>
            <p>波形と音声を確認して回答してください。</p>
          </div>
        </div>
        {form.candidateQuestions.map((question, questionIndex) => (
          <fieldset className="question-fieldset" key={question.id}>
            <legend>
              <span>{questionIndex + 1}</span>
              {question.prompt}
            </legend>
            <div className="choice-grid">
              {question.choices.map((choice) => {
                const inputId = `${selectedCandidate.id}-${question.id}-${choice.value}`
                return (
                  <label className="choice" key={choice.value} htmlFor={inputId}>
                    <input
                      id={inputId}
                      type="radio"
                      name={`${selectedCandidate.id}-${question.id}`}
                      value={choice.value}
                      checked={
                        draft.candidateAnswers[selectedCandidate.id]?.[question.id] ===
                        choice.value
                      }
                      onChange={() =>
                        onChange(
                          answerCandidateQuestion(
                            draft,
                            selectedCandidate.id,
                            question.id,
                            choice.value,
                          ),
                        )
                      }
                    />
                    <span>{choice.label}</span>
                  </label>
                )
              })}
            </div>
          </fieldset>
        ))}
      </div>

      <fieldset className="question-fieldset comparison-fieldset">
        <legend>{form.comparison.prompt}</legend>
        <div className="comparison-choices">
          {candidates.map((candidate) => (
            <ComparisonChoice
              key={candidate.id}
              label={`候補${candidate.id}が最も自然`}
              value={{ outcome: 'candidate', candidateId: candidate.id }}
              checked={
                draft.comparison?.outcome === 'candidate' &&
                draft.comparison.candidateId === candidate.id
              }
              onChange={(answer) => onChange(answerComparison(draft, answer))}
            />
          ))}
          {form.comparison.allowIndistinguishable && candidates.length > 1 && (
            <ComparisonChoice
              label="聴感上の差を判断できない"
              value={{ outcome: 'indistinguishable' }}
              checked={draft.comparison?.outcome === 'indistinguishable'}
              onChange={(answer) => onChange(answerComparison(draft, answer))}
            />
          )}
          {form.comparison.allowNone && (
            <ComparisonChoice
              label="どの候補も不適切"
              value={{ outcome: 'none' }}
              checked={draft.comparison?.outcome === 'none'}
              onChange={(answer) => onChange(answerComparison(draft, answer))}
            />
          )}
        </div>
      </fieldset>
    </section>
  )
}

function ComparisonChoice({
  label,
  value,
  checked,
  onChange,
}: {
  label: string
  value: ComparisonAnswer
  checked: boolean
  onChange: (answer: ComparisonAnswer) => void
}) {
  const valueKey =
    value.outcome === 'candidate'
      ? `candidate-${value.candidateId}`
      : value.outcome

  return (
    <label className="comparison-choice">
      <input
        type="radio"
        name="comparison"
        value={valueKey}
        checked={checked}
        onChange={() => onChange(value)}
      />
      <span>{label}</span>
    </label>
  )
}

