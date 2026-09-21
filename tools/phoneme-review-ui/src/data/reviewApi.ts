import {
  parseReviewRecord,
  type ReviewRecord,
  type ReviewSubmission,
} from '../domain/reviewRecord.ts'

const REVIEWS_API_PATH = '/api/reviews'

export class ReviewApiError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'ReviewApiError'
  }
}

export async function loadLatestReviewRecords(
  datasetId: string,
  datasetVersion: string,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<ReviewRecord[]> {
  const query = new URLSearchParams({ datasetId, datasetVersion })
  const response = await request(
    `${REVIEWS_API_PATH}?${query.toString()}`,
    { signal },
    fetcher,
  )
  const body = objectAt(await response.json(), '$')
  if (!Array.isArray(body.records)) {
    throw new ReviewApiError('保存済み回答の応答形式が正しくありません。')
  }
  try {
    return body.records.map((record) => parseReviewRecord(record))
  } catch (error) {
    throw new ReviewApiError('保存済み回答の応答形式が正しくありません。', {
      cause: error,
    })
  }
}

export async function saveReviewSubmission(
  submission: ReviewSubmission,
  fetcher: typeof fetch = fetch,
): Promise<ReviewRecord> {
  const response = await request(
    REVIEWS_API_PATH,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(submission),
    },
    fetcher,
  )
  const body = objectAt(await response.json(), '$')
  try {
    return parseReviewRecord(body.record)
  } catch (error) {
    throw new ReviewApiError('回答保存の応答形式が正しくありません。', {
      cause: error,
    })
  }
}

async function request(
  url: string,
  init: RequestInit,
  fetcher: typeof fetch,
): Promise<Response> {
  let response: Response
  try {
    response = await fetcher(url, init)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new ReviewApiError('回答保存APIへ接続できませんでした。', {
      cause: error,
    })
  }

  if (!response.ok) {
    let detail = ''
    try {
      const body = objectAt(await response.json(), '$')
      if (typeof body.error === 'string') {
        detail = `: ${body.error}`
      }
    } catch {
      // The status code still provides a useful error when the body is invalid.
    }
    throw new ReviewApiError(
      `回答保存APIでエラーが発生しました（HTTP ${response.status}）${detail}`,
    )
  }

  return response
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ReviewApiError(`${path}: must be an object`)
  }
  return value as Record<string, unknown>
}

