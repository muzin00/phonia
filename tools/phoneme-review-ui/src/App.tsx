import { useEffect, useMemo, useRef, useState } from 'react'

import { EvaluationForm } from './components/EvaluationForm.tsx'
import { WaveformPanel } from './components/WaveformPanel.tsx'
import { loadReviewDataset } from './data/loadReviewDataset.ts'
import {
  loadLatestReviewRecords,
  saveReviewSubmission,
} from './data/reviewApi.ts'
import {
  createEmptyReviewDraft,
  getReviewProgress,
  type ReviewDraft,
} from './domain/reviewDraft.ts'
import {
  loadLocalReviewState,
  saveLocalReviewState,
} from './domain/reviewLocalStorage.ts'
import {
  audioFileOptions,
  nextReviewIndex,
} from './domain/reviewNavigation.ts'
import {
  createReviewSubmission,
  reviewRecordToDraft,
} from './domain/reviewPersistence.ts'
import type {
  ReviewCandidate,
  ReviewDataset,
  ReviewItem,
} from './domain/reviewDataset.ts'

const DEFAULT_DATASET_URL = '/review-dataset.json'

type LoadState =
  | { status: 'loading' }
  | { status: 'loaded'; dataset: ReviewDataset }
  | { status: 'error'; message: string }

type ItemFileState = 'dirty' | 'saved'

type FileExportState =
  | { status: 'idle' | 'saving' }
  | { status: 'saved'; count: number; savedAt: string }
  | { status: 'error'; message: string }

export default function App() {
  const datasetUrl = useMemo(() => getDatasetUrl(), [])
  const [reloadCount, setReloadCount] = useState(0)
  const [loadState, setLoadState] = useState<LoadState>({ status: 'loading' })
  const [currentIndex, setCurrentIndex] = useState(0)
  const [drafts, setDrafts] = useState<Record<string, ReviewDraft>>({})
  const draftsRef = useRef(drafts)
  const [itemFileStates, setItemFileStates] = useState<
    Record<string, ItemFileState>
  >({})
  const [fileExportState, setFileExportState] = useState<FileExportState>({
    status: 'idle',
  })

  useEffect(() => {
    draftsRef.current = drafts
  }, [drafts])

  useEffect(() => {
    const controller = new AbortController()

    void (async () => {
      const dataset = await loadReviewDataset(datasetUrl, controller.signal)
      const records = await loadLatestReviewRecords(
        dataset.datasetId,
        dataset.datasetVersion,
        controller.signal,
      )
      const fileDrafts: Record<string, ReviewDraft> = {}
      const loadedFileStates: Record<string, ItemFileState> = {}
      records.forEach((record) => {
        fileDrafts[record.itemId] = reviewRecordToDraft(record)
        loadedFileStates[record.itemId] = 'saved'
      })

      const localState = safelyLoadLocalReviewState(dataset)
      const loadedDrafts = { ...fileDrafts }
      if (localState !== null) {
        Object.entries(localState.drafts).forEach(([itemId, draft]) => {
          if (!dataset.items.some((item) => item.id === itemId)) {
            return
          }
          loadedDrafts[itemId] = draft
          loadedFileStates[itemId] = draftsEqual(draft, fileDrafts[itemId])
            ? 'saved'
            : 'dirty'
        })
      }

      if (!controller.signal.aborted) {
        setCurrentIndex(
          Math.min(
            dataset.items.length - 1,
            localState?.currentIndex ?? 0,
          ),
        )
        setDrafts(loadedDrafts)
        setItemFileStates(loadedFileStates)
        setFileExportState({ status: 'idle' })
        setLoadState({ status: 'loaded', dataset })
      }
    })()
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        setLoadState({ status: 'error', message: errorMessage(error) })
      })

    return () => controller.abort()
  }, [datasetUrl, reloadCount])

  useEffect(() => {
    if (loadState.status !== 'loaded') {
      return
    }
    safelySaveLocalReviewState(
      loadState.dataset,
      currentIndex,
      drafts,
    )
  }, [currentIndex, drafts, loadState])

  if (loadState.status === 'loading') {
    return <LoadingScreen datasetUrl={datasetUrl} />
  }

  if (loadState.status === 'error') {
    return (
      <ErrorScreen
        datasetUrl={datasetUrl}
        message={loadState.message}
        onRetry={() => {
          setLoadState({ status: 'loading' })
          setReloadCount((count) => count + 1)
        }}
      />
    )
  }

  const { dataset } = loadState
  const item = dataset.items[currentIndex]
  if (item === undefined) {
    return <ErrorScreen datasetUrl={datasetUrl} message="表示する項目がありません。" />
  }

  const exportableItems = dataset.items.filter((candidateItem) => {
    const draft = drafts[candidateItem.id]
    return (
      draft !== undefined &&
      itemFileStates[candidateItem.id] !== 'saved' &&
      getReviewProgress(dataset.form, candidateItem.candidates, draft).completed
    )
  })

  return (
    <ReviewScreen
      key={item.id}
      dataset={dataset}
      item={item}
      currentIndex={currentIndex}
      draft={drafts[item.id] ?? createEmptyReviewDraft()}
      onDraftChange={(draft) => {
        const previousDraft = drafts[item.id] ?? createEmptyReviewDraft()
        const wasCompleted = getReviewProgress(
          dataset.form,
          item.candidates,
          previousDraft,
        ).completed
        const isCompleted = getReviewProgress(
          dataset.form,
          item.candidates,
          draft,
        ).completed
        setDrafts((currentDrafts) => ({
          ...currentDrafts,
          [item.id]: draft,
        }))
        setItemFileStates((currentStates) => ({
          ...currentStates,
          [item.id]: 'dirty',
        }))
        setFileExportState({ status: 'idle' })
        if (!wasCompleted && isCompleted) {
          setCurrentIndex(nextReviewIndex(currentIndex, dataset.items.length))
        }
      }}
      fileExportState={fileExportState}
      exportableCount={exportableItems.length}
      onExport={() => {
        const draftsAtExport = Object.fromEntries(
          exportableItems.map((exportItem) => [
            exportItem.id,
            drafts[exportItem.id]!,
          ]),
        )
        setFileExportState({ status: 'saving' })
        void exportReviewItems(
          dataset,
          exportableItems,
          draftsAtExport,
          (exportedItem) => {
            setItemFileStates((currentStates) => {
              if (
                !draftsEqual(
                  draftsAtExport[exportedItem.id],
                  draftsRef.current[exportedItem.id],
                )
              ) {
                return currentStates
              }
              return { ...currentStates, [exportedItem.id]: 'saved' }
            })
          },
        )
          .then(() => {
            setFileExportState({
              status: 'saved',
              count: exportableItems.length,
              savedAt: new Date().toISOString(),
            })
          })
          .catch((error: unknown) => {
            setFileExportState({
              status: 'error',
              message: errorMessage(error),
            })
          })
      }}
      onPrevious={() => setCurrentIndex((index) => Math.max(0, index - 1))}
      onNext={() =>
        setCurrentIndex((index) => Math.min(dataset.items.length - 1, index + 1))
      }
      onSelectAudioFile={(firstItemIndex) => setCurrentIndex(firstItemIndex)}
    />
  )
}

function ReviewScreen({
  dataset,
  item,
  currentIndex,
  draft,
  onDraftChange,
  fileExportState,
  exportableCount,
  onExport,
  onPrevious,
  onNext,
  onSelectAudioFile,
}: {
  dataset: ReviewDataset
  item: ReviewItem
  currentIndex: number
  draft: ReviewDraft
  onDraftChange: (draft: ReviewDraft) => void
  fileExportState: FileExportState
  exportableCount: number
  onExport: () => void
  onPrevious: () => void
  onNext: () => void
  onSelectAudioFile: (firstItemIndex: number) => void
}) {
  const availableCandidates = item.candidates.filter(
    (candidate): candidate is Extract<
      ReviewCandidate,
      { status: 'available' }
    > => candidate.status === 'available',
  )
  const [selectedCandidateId, setSelectedCandidateId] = useState(
    availableCandidates[0]?.id,
  )
  const selectedCandidate = availableCandidates.find(
    (candidate) => candidate.id === selectedCandidateId,
  )
  const audioFiles = audioFileOptions(dataset.items)

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Phonia · Review</p>
          <h1>{dataset.title}</h1>
        </div>
        <div className="header-controls">
          <label className="audio-file-selector">
            <span>音声ファイル</span>
            <select
              value={item.utterance.audioUrl}
              onChange={(event) => {
                const option = audioFiles.find(
                  (audioFile) => audioFile.audioUrl === event.target.value,
                )
                if (option !== undefined) {
                  onSelectAudioFile(option.firstItemIndex)
                }
              }}
            >
              {audioFiles.map((audioFile) => (
                <option key={audioFile.audioUrl} value={audioFile.audioUrl}>
                  {audioFile.label}（{audioFile.itemCount}区間）
                </option>
              ))}
            </select>
          </label>
          <div className="file-export-control">
            <button
              type="button"
              className="button button-secondary"
              disabled={
                exportableCount === 0 || fileExportState.status === 'saving'
              }
              onClick={onExport}
            >
              {fileExportState.status === 'saving'
                ? 'JSONLへ保存中…'
                : `JSONLへ保存（${exportableCount}件）`}
            </button>
            <FileExportStatus state={fileExportState} />
          </div>
          <p className="progress" aria-label="レビュー進捗">
            <strong>{currentIndex + 1}</strong>
            <span>/ {dataset.items.length}</span>
          </p>
        </div>
      </header>

      <section className="context-card" aria-labelledby="utterance-heading">
        <div className="section-heading">
          <div>
            <p className="section-label">Utterance</p>
            <h2 id="utterance-heading">発話コンテキスト</h2>
          </div>
          <code>{item.utterance.id}</code>
        </div>
        <blockquote>{item.utterance.text}</blockquote>
        <dl className="target-details">
          <div>
            <dt>対象音素</dt>
            <dd lang="en">/{item.target.label}/</dd>
          </div>
          <div>
            <dt>発話内位置</dt>
            <dd>{item.target.index + 1}</dd>
          </div>
          <div>
            <dt>期待単位数</dt>
            <dd>{item.target.unitCount}</dd>
          </div>
          <div>
            <dt>利用可能候補</dt>
            <dd>
              {availableCandidates.length} / {item.candidates.length}
            </dd>
          </div>
        </dl>
        {item.target.tags.length > 0 && (
          <ul className="tag-list" aria-label="対象属性">
            {item.target.tags.map((tag) => (
              <li key={tag}>{tag}</li>
            ))}
          </ul>
        )}
      </section>

      <section className="candidate-section" aria-labelledby="candidates-heading">
        <div className="section-heading">
          <div>
            <p className="section-label">Candidates</p>
            <h2 id="candidates-heading">候補区間</h2>
          </div>
          <p className="supporting-text">モデル名は匿名化されています</p>
        </div>
        {selectedCandidate !== undefined && (
          <WaveformPanel
            audioUrl={item.utterance.audioUrl}
            candidate={selectedCandidate}
            contextPaddingSec={dataset.playback.contextPaddingSec}
          />
        )}
        <div className="candidate-grid">
          {item.candidates.map((candidate) => (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              selected={candidate.id === selectedCandidateId}
              onSelect={() => setSelectedCandidateId(candidate.id)}
            />
          ))}
        </div>
      </section>

      {selectedCandidateId !== undefined && (
        <EvaluationForm
          form={dataset.form}
          candidates={availableCandidates}
          selectedCandidateId={selectedCandidateId}
          draft={draft}
          onChange={onDraftChange}
        />
      )}

      <footer className="review-navigation">
        <button
          type="button"
          className="button button-secondary"
          disabled={currentIndex === 0}
          onClick={onPrevious}
        >
          前へ
        </button>
        <p>
          データセット {dataset.datasetId}@{dataset.datasetVersion}
        </p>
        <button
          type="button"
          className="button button-primary"
          disabled={currentIndex === dataset.items.length - 1}
          onClick={onNext}
        >
          次へ
        </button>
      </footer>
    </main>
  )
}

function FileExportStatus({ state }: { state: FileExportState }) {
  if (state.status === 'saving') {
    return <span aria-live="polite">書き込み中</span>
  }
  if (state.status === 'saved') {
    return (
      <span aria-live="polite">
        {state.count}件を保存（{new Date(state.savedAt).toLocaleTimeString('ja-JP')}）
      </span>
    )
  }
  if (state.status === 'error') {
    return (
      <span className="file-export-error" role="alert">
        {state.message}
      </span>
    )
  }
  return <span>完了回答を任意のタイミングで保存</span>
}

function CandidateCard({
  candidate,
  selected,
  onSelect,
}: {
  candidate: ReviewCandidate
  selected: boolean
  onSelect: () => void
}) {
  return (
    <article
      className={`candidate-card candidate-card-${candidate.status}${selected ? ' candidate-card-selected' : ''}`}
      aria-label={`候補${candidate.id}`}
    >
      <div className="candidate-title">
        {candidate.status === 'available' ? (
          <button
            type="button"
            className="candidate-id candidate-id-button"
            aria-label={`候補${candidate.id}を波形に表示`}
            aria-pressed={selected}
            onClick={onSelect}
          >
            {candidate.id}
          </button>
        ) : (
          <span className="candidate-id">{candidate.id}</span>
        )}
        <span className={`status status-${candidate.status}`}>
          {selected
            ? '選択中'
            : candidate.status === 'available'
              ? '利用可能'
              : '区間なし'}
        </span>
      </div>
      {candidate.status === 'available' ? (
        <dl className="segment-times">
          <div>
            <dt>開始</dt>
            <dd>{formatSeconds(candidate.segment.startSec)}</dd>
          </div>
          <div>
            <dt>終了</dt>
            <dd>{formatSeconds(candidate.segment.endSec)}</dd>
          </div>
          <div>
            <dt>長さ</dt>
            <dd>
              {formatSeconds(candidate.segment.endSec - candidate.segment.startSec)}
            </dd>
          </div>
        </dl>
      ) : (
        <p className="missing-message">この方式は対象区間を出力しませんでした。</p>
      )}
    </article>
  )
}

function LoadingScreen({ datasetUrl }: { datasetUrl: string }) {
  return (
    <main className="state-screen" aria-busy="true">
      <span className="loading-indicator" aria-hidden="true" />
      <p className="eyebrow">Phonia · Review</p>
      <h1>レビューデータを読み込んでいます</h1>
      <code>{datasetUrl}</code>
    </main>
  )
}

function ErrorScreen({
  datasetUrl,
  message,
  onRetry,
}: {
  datasetUrl: string
  message: string
  onRetry?: () => void
}) {
  return (
    <main className="state-screen state-screen-error" role="alert">
      <p className="eyebrow">読み込みエラー</p>
      <h1>データセットを表示できません</h1>
      <p>{message}</p>
      <code>{datasetUrl}</code>
      {onRetry !== undefined && (
        <button type="button" className="button button-primary" onClick={onRetry}>
          再読み込み
        </button>
      )}
    </main>
  )
}

function getDatasetUrl(): string {
  const url = new URL(window.location.href)
  return url.searchParams.get('dataset')?.trim() || DEFAULT_DATASET_URL
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '不明なエラーが発生しました。'
}

function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(3)} s`
}

async function exportReviewItems(
  dataset: ReviewDataset,
  items: ReviewItem[],
  drafts: Record<string, ReviewDraft>,
  onItemSaved: (item: ReviewItem) => void,
): Promise<void> {
  for (const item of items) {
    const draft = drafts[item.id]
    if (draft === undefined) {
      continue
    }
    await saveReviewSubmission(createReviewSubmission(dataset, item, draft))
    onItemSaved(item)
  }
}

function safelyLoadLocalReviewState(dataset: ReviewDataset) {
  try {
    return loadLocalReviewState(
      window.localStorage,
      dataset.datasetId,
      dataset.datasetVersion,
    )
  } catch {
    return null
  }
}

function safelySaveLocalReviewState(
  dataset: ReviewDataset,
  currentIndex: number,
  drafts: Record<string, ReviewDraft>,
): void {
  try {
    saveLocalReviewState(
      window.localStorage,
      dataset.datasetId,
      dataset.datasetVersion,
      { currentIndex, drafts },
    )
  } catch {
    // Continue in memory when browser storage is unavailable.
  }
}

function draftsEqual(
  left: ReviewDraft | undefined,
  right: ReviewDraft | undefined,
): boolean {
  return JSON.stringify(left) === JSON.stringify(right)
}
