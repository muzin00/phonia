import { parseReviewDataset } from '../domain/parseReviewDataset.ts'
import type { ReviewDataset } from '../domain/reviewDataset.ts'

export class ReviewDatasetLoadError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options)
    this.name = 'ReviewDatasetLoadError'
  }
}

export async function loadReviewDataset(
  url: string,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<ReviewDataset> {
  let response: Response

  try {
    response = await fetcher(url, { signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new ReviewDatasetLoadError(
      `レビューデータを取得できませんでした: ${url}`,
      { cause: error },
    )
  }

  if (!response.ok) {
    throw new ReviewDatasetLoadError(
      `レビューデータの取得に失敗しました（HTTP ${response.status}）: ${url}`,
    )
  }

  let value: unknown
  try {
    value = await response.json()
  } catch (error) {
    throw new ReviewDatasetLoadError(
      `レビューデータがJSONではありません: ${url}`,
      { cause: error },
    )
  }

  try {
    return parseReviewDataset(value)
  } catch (error) {
    const detail = error instanceof Error ? `（${error.message}）` : ''
    throw new ReviewDatasetLoadError(
      `レビューデータの形式が正しくありません${detail}: ${url}`,
      { cause: error },
    )
  }
}
