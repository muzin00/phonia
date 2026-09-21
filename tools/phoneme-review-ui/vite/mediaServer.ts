import { createReadStream } from 'node:fs'
import { stat } from 'node:fs/promises'
import { extname, relative, resolve } from 'node:path'

import type { Plugin, PreviewServer, ViteDevServer } from 'vite'

const MEDIA_PREFIX = '/media/'
const MIME_TYPES: Record<string, string> = {
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.wav': 'audio/wav',
}

export function mediaServerPlugin(mediaRoot: string): Plugin {
  const absoluteRoot = resolve(mediaRoot)

  const installMiddleware = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use((request, response, next) => {
      const pathname = new URL(request.url ?? '/', 'http://localhost').pathname
      if (!pathname.startsWith(MEDIA_PREFIX)) {
        next()
        return
      }

      const filePath = resolveMediaFile(absoluteRoot, pathname)
      if (filePath === null) {
        response.statusCode = 400
        response.end('Invalid media path')
        return
      }

      void stat(filePath)
        .then((file) => {
          if (!file.isFile()) {
            response.statusCode = 404
            response.end('Media file not found')
            return
          }

          response.statusCode = 200
          response.setHeader(
            'Content-Type',
            MIME_TYPES[extname(filePath).toLowerCase()] ?? 'application/octet-stream',
          )
          response.setHeader('Cache-Control', 'no-store')
          response.setHeader('Accept-Ranges', 'bytes')

          const requestedRange = request.headers.range
          const byteRange =
            requestedRange === undefined
              ? null
              : parseByteRange(requestedRange, file.size)

          if (requestedRange !== undefined && byteRange === null) {
            response.statusCode = 416
            response.setHeader('Content-Range', `bytes */${file.size}`)
            response.end()
            return
          }

          if (byteRange !== null) {
            response.statusCode = 206
            response.setHeader(
              'Content-Range',
              `bytes ${byteRange.start}-${byteRange.end}/${file.size}`,
            )
            response.setHeader('Content-Length', byteRange.end - byteRange.start + 1)
          } else {
            response.setHeader('Content-Length', file.size)
          }

          if (request.method === 'HEAD') {
            response.end()
            return
          }

          createReadStream(filePath, byteRange ?? undefined).pipe(response)
        })
        .catch(() => {
          response.statusCode = 404
          response.end('Media file not found')
        })
    })
  }

  return {
    name: 'phonia-media-server',
    configureServer: installMiddleware,
    configurePreviewServer: installMiddleware,
  }
}

export function parseByteRange(
  rangeHeader: string,
  fileSize: number,
): { start: number; end: number } | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader)
  if (match === null || fileSize <= 0) {
    return null
  }

  const [, startText, endText] = match
  if (startText === '' && endText === '') {
    return null
  }

  if (startText === '') {
    const suffixLength = Number(endText)
    if (!Number.isInteger(suffixLength) || suffixLength <= 0) {
      return null
    }
    return {
      start: Math.max(0, fileSize - suffixLength),
      end: fileSize - 1,
    }
  }

  const start = Number(startText)
  const requestedEnd = endText === '' ? fileSize - 1 : Number(endText)
  if (
    !Number.isInteger(start) ||
    !Number.isInteger(requestedEnd) ||
    start < 0 ||
    start >= fileSize ||
    requestedEnd < start
  ) {
    return null
  }

  return {
    start,
    end: Math.min(requestedEnd, fileSize - 1),
  }
}

export function resolveMediaFile(
  mediaRoot: string,
  requestPathname: string,
): string | null {
  let decodedPath: string
  try {
    decodedPath = decodeURIComponent(requestPathname.slice(MEDIA_PREFIX.length))
  } catch {
    return null
  }

  const filePath = resolve(mediaRoot, decodedPath)
  const relativePath = relative(mediaRoot, filePath)
  if (
    decodedPath.length === 0 ||
    relativePath.startsWith('..') ||
    relativePath === '' ||
    extname(filePath).toLowerCase() in MIME_TYPES === false
  ) {
    return null
  }

  return filePath
}
