import {
  answerCandidateQuestion,
  answerReviewStatus,
  getReviewProgress,
  type ReviewDraft,
} from '../domain/reviewDraft.ts'
import type {
  ReviewCandidate,
  ReviewForm,
  ReviewStatus,
} from '../domain/reviewDataset.ts'

type AvailableCandidate = Extract<ReviewCandidate, { status: 'available' }>

export function EvaluationForm({
  form,
  candidates,
  selectedCandidateId,
  draft,
  onChange,
}: {
  form: ReviewForm
  candidates: AvailableCandidate[]
  selectedCandidateId: string
  draft: ReviewDraft
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

      <div className="candidate-questions">
        <div className="question-group-heading">
          <span className="candidate-id">{selectedCandidate.id}</span>
          <div>
            <h3>母音区間の評価</h3>
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

      <fieldset className="question-fieldset review-status-fieldset">
        <legend>{form.reviewStatus.prompt}</legend>
        <div className="review-status-choices">
          {form.reviewStatus.choices.map((choice) => (
            <ReviewStatusChoice
              key={choice.value}
              label={choice.label}
              value={choice.value}
              checked={draft.reviewStatus === choice.value}
              onChange={(answer) =>
                onChange(answerReviewStatus(draft, answer))
              }
            />
          ))}
        </div>
      </fieldset>

      <div className="local-save-panel" aria-live="polite">
        <p className="save-title">ブラウザへ自動保存</p>
        <p className="save-message">
          {progress.completed
            ? '回答済みです。次の区間へ自動的に進みます。'
            : '選択内容はこのブラウザに自動保存されています。'}
        </p>
      </div>
    </section>
  )
}

function ReviewStatusChoice({
  label,
  value,
  checked,
  onChange,
}: {
  label: string
  value: ReviewStatus
  checked: boolean
  onChange: (answer: ReviewStatus) => void
}) {
  return (
    <label className="review-status-choice">
      <input
        type="radio"
        name="review-status"
        value={value}
        checked={checked}
        onChange={() => onChange(value)}
      />
      <span>{label}</span>
    </label>
  )
}
