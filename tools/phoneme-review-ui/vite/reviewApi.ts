import { resolve } from 'node:path'

import type { Plugin, PreviewServer, ViteDevServer } from 'vite'

import {
  parseReviewSubmission,
  type ReviewRecord,
} from '../src/domain/reviewRecord.ts'
import {
  appendReviewSubmission,
  readReviewRecords,
  selectLatestReviewRecords,
} from './reviewStore.ts'

const API_PATH = '/api/reviews'
const MAX_REQUEST_BYTES = 1024 * 1024

export function reviewApiPlugin(reviewOutputPath: string): Plugin {
  const absoluteOutputPath = resolve(reviewOutputPath)
  let writeQueue = Promise.resolve()

  const enqueueWrite = (operation: () => Promise<ReviewRecord>) => {
    const result = writeQueue.then(operation)
    writeQueue = result.then(
      () => undefined,
      () => undefined,
    )
    return result
  }

  const installMiddleware = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use((request, response, next) => {
      const url = new URL(request.url ?? '/', 'http://localhost')
      if (url.pathname !== API_PATH) {
        next()
        return
      }

      if (request.method === 'GET') {
        const datasetId = url.searchParams.get('datasetId')?.trim()
        const datasetVersion = url.searchParams.get('datasetVersion')?.trim()
        if (!datasetId || !datasetVersion) {
          sendJson(response, 400, {
            error: 'datasetId and datasetVersion are required',
          })
          return
        }

        void readReviewRecords(absoluteOutputPath)
          .then((records) =>
            sendJson(response, 200, {
              records: selectLatestReviewRecords(
                records,
                datasetId,
                datasetVersion,
              ),
            }),
          )
          .catch((error: unknown) => {
            sendServerError(response, error)
          })
        return
      }

      if (request.method === 'POST') {
        void readJsonBody(request)
          .then((value) => parseReviewSubmission(value))
          .then((submission) =>
            enqueueWrite(() =>
              appendReviewSubmission(absoluteOutputPath, submission),
            ),
          )
          .then((record) => sendJson(response, 201, { record }))
          .catch((error: unknown) => {
            if (error instanceof RequestBodyError) {
              sendJson(response, error.statusCode, { error: error.message })
              return
            }
            if (error instanceof Error && error.name === 'InvalidReviewRecordError') {
              sendJson(response, 400, { error: error.message })
              return
            }
            sendServerError(response, error)
          })
        return
      }

      response.setHeader('Allow', 'GET, POST')
      sendJson(response, 405, { error: 'Method not allowed' })
    })
  }

  return {
    name: 'phonia-review-api',
    configureServer: installMiddleware,
    configurePreviewServer: installMiddleware,
  }
}

async function readJsonBody(request: NodeJS.ReadableStream): Promise<unknown> {
  const chunks: Buffer[] = []
  let totalBytes = 0

  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)
    totalBytes += buffer.length
    if (totalBytes > MAX_REQUEST_BYTES) {
      throw new RequestBodyError(413, 'Request body is too large')
    }
    chunks.push(buffer)
  }

  try {
    return JSON.parse(Buffer.concat(chunks).toString('utf8')) as unknown
  } catch (error) {
    throw new RequestBodyError(400, 'Request body must be valid JSON', {
      cause: error,
    })
  }
}

function sendJson(
  response: NodeJS.WritableStream & {
    statusCode: number
    setHeader(name: string, value: string): void
    end(body?: string): void
  },
  statusCode: number,
  body: unknown,
) {
  response.statusCode = statusCode
  response.setHeader('Content-Type', 'application/json; charset=utf-8')
  response.setHeader('Cache-Control', 'no-store')
  response.end(`${JSON.stringify(body)}\n`)
}

function sendServerError(
  response: Parameters<typeof sendJson>[0],
  error: unknown,
) {
  console.error(error)
  sendJson(response, 500, { error: 'Failed to access review records' })
}

class RequestBodyError extends Error {
  readonly statusCode: number

  constructor(
    statusCode: number,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options)
    this.name = 'RequestBodyError'
    this.statusCode = statusCode
  }
}
