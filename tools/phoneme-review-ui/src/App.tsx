import { useEffect, useMemo, useState } from 'react'

import { loadReviewDataset } from './data/loadReviewDataset.ts'
import type {
  ReviewCandidate,
  ReviewDataset,
  ReviewItem,
} from './domain/reviewDataset.ts'

const DEFAULT_DATASET_URL = '/examples/review-dataset.json'

type LoadState =
  | { status: 'loading' }
  | { status: 'loaded'; dataset: ReviewDataset }
  | { status: 'error'; message: string }

export default function App() {
  const datasetUrl = useMemo(() => getDatasetUrl(), [])
  const [reloadCount, setReloadCount] = useState(0)
  const [loadState, setLoadState] = useState<LoadState>({ status: 'loading' })
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    const controller = new AbortController()

    void loadReviewDataset(datasetUrl, controller.signal)
      .then((dataset) => {
        setCurrentIndex(0)
        setLoadState({ status: 'loaded', dataset })
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return
        }
        setLoadState({ status: 'error', message: errorMessage(error) })
      })

    return () => controller.abort()
  }, [datasetUrl, reloadCount])

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

  return (
    <ReviewScreen
      dataset={dataset}
      item={item}
      currentIndex={currentIndex}
      onPrevious={() => setCurrentIndex((index) => Math.max(0, index - 1))}
      onNext={() =>
        setCurrentIndex((index) => Math.min(dataset.items.length - 1, index + 1))
      }
    />
  )
}

function ReviewScreen({
  dataset,
  item,
  currentIndex,
  onPrevious,
  onNext,
}: {
  dataset: ReviewDataset
  item: ReviewItem
  currentIndex: number
  onPrevious: () => void
  onNext: () => void
}) {
  const availableCount = item.candidates.filter(
    (candidate) => candidate.status === 'available',
  ).length

  return (
    <main className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Phonia · Review</p>
          <h1>{dataset.title}</h1>
        </div>
        <p className="progress" aria-label="レビュー進捗">
          <strong>{currentIndex + 1}</strong>
          <span>/ {dataset.items.length}</span>
        </p>
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
              {availableCount} / {item.candidates.length}
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
        <div className="candidate-grid">
          {item.candidates.map((candidate) => (
            <CandidateCard key={candidate.id} candidate={candidate} />
          ))}
        </div>
      </section>

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

function CandidateCard({ candidate }: { candidate: ReviewCandidate }) {
  return (
    <article
      className={`candidate-card candidate-card-${candidate.status}`}
      aria-label={`候補${candidate.id}`}
    >
      <div className="candidate-title">
        <span className="candidate-id">{candidate.id}</span>
        <span className={`status status-${candidate.status}`}>
          {candidate.status === 'available' ? '利用可能' : '区間なし'}
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
