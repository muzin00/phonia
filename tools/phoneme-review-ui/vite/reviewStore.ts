import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import {
  parseReviewRecord,
  type ReviewRecord,
  type ReviewSubmission,
} from '../src/domain/reviewRecord.ts'

export async function readReviewRecords(filePath: string): Promise<ReviewRecord[]> {
  let source: string
  try {
    source = await readFile(filePath, 'utf8')
  } catch (error) {
    if (isErrorWithCode(error) && error.code === 'ENOENT') {
      return []
    }
    throw error
  }

  return source
    .split(/\r?\n/)
    .map((line, index) => ({ line, lineNumber: index + 1 }))
    .filter(({ line }) => line.trim().length > 0)
    .map(({ line, lineNumber }) => {
      try {
        return parseReviewRecord(JSON.parse(line) as unknown)
      } catch (error) {
        const detail = error instanceof Error ? error.message : String(error)
        throw new Error(`Invalid review record at line ${lineNumber}: ${detail}`, {
          cause: error,
        })
      }
    })
}

export function selectLatestReviewRecords(
  records: ReviewRecord[],
  datasetId: string,
  datasetVersion: string,
): ReviewRecord[] {
  const latestByItem = new Map<string, ReviewRecord>()

  records.forEach((record) => {
    if (
      record.datasetId !== datasetId ||
      record.datasetVersion !== datasetVersion
    ) {
      return
    }
    const current = latestByItem.get(record.itemId)
    if (current === undefined || record.revision > current.revision) {
      latestByItem.set(record.itemId, record)
    }
  })

  return [...latestByItem.values()].sort((left, right) =>
    left.itemId.localeCompare(right.itemId),
  )
}

export async function appendReviewSubmission(
  filePath: string,
  submission: ReviewSubmission,
  recordedAt: string = new Date().toISOString(),
): Promise<ReviewRecord> {
  const records = await readReviewRecords(filePath)
  const revision =
    records.reduce(
      (maximum, record) =>
        record.datasetId === submission.datasetId &&
        record.datasetVersion === submission.datasetVersion &&
        record.itemId === submission.itemId
          ? Math.max(maximum, record.revision)
          : maximum,
      0,
    ) + 1
  const record: ReviewRecord = {
    ...submission,
    revision,
    recordedAt,
  }

  await mkdir(dirname(filePath), { recursive: true })
  await appendFile(filePath, `${JSON.stringify(record)}\n`, 'utf8')
  return record
}

function isErrorWithCode(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error
}

