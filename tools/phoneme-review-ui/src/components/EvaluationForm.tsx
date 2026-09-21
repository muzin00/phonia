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

export type ReviewSaveState =
  | { status: 'idle' | 'dirty' | 'saving' }
  | { status: 'saved'; revision: number; recordedAt: string }
  | { status: 'error'; message: string }

export function EvaluationForm({
  form,
  candidates,
  selectedCandidateId,
  draft,
  saveState,
  onSelectCandidate,
  onChange,
  onSave,
}: {
  form: ReviewForm
  candidates: AvailableCandidate[]
  selectedCandidateId: string
  draft: ReviewDraft
  saveState: ReviewSaveState
  onSelectCandidate: (candidateId: string) => void
  onChange: (draft: ReviewDraft) => void
  onSave: () => void
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

      <div className="save-panel">
        <div aria-live="polite">
          <p className="save-title">回答の保存</p>
          <SaveStatus state={saveState} completed={progress.completed} />
        </div>
        <button
          type="button"
          className="button button-primary save-button"
          disabled={
            !progress.completed ||
            saveState.status === 'saving' ||
            saveState.status === 'saved'
          }
          onClick={onSave}
        >
          {saveState.status === 'saving'
            ? '保存中…'
            : saveState.status === 'dirty'
              ? '変更を保存'
              : '回答を保存'}
        </button>
      </div>
    </section>
  )
}

function SaveStatus({
  state,
  completed,
}: {
  state: ReviewSaveState
  completed: boolean
}) {
  if (state.status === 'saved') {
    return (
      <p className="save-message save-message-success">
        リビジョン{state.revision}として保存しました（
        {new Date(state.recordedAt).toLocaleString('ja-JP')}）
      </p>
    )
  }
  if (state.status === 'error') {
    return (
      <p className="save-message save-message-error" role="alert">
        {state.message}
      </p>
    )
  }
  if (state.status === 'saving') {
    return <p className="save-message">JSONLへ追記しています。</p>
  }
  if (!completed) {
    return <p className="save-message">すべての設問へ回答すると保存できます。</p>
  }
  return <p className="save-message">未保存の回答があります。</p>
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
