import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import type { Plugin, PreviewServer, ViteDevServer } from 'vite'

const DATASET_PATH = '/review-dataset.json'

export function datasetServerPlugin(datasetPath: string): Plugin {
  const absoluteDatasetPath = resolve(datasetPath)

  const installMiddleware = (server: ViteDevServer | PreviewServer) => {
    server.middlewares.use((request, response, next) => {
      const pathname = new URL(request.url ?? '/', 'http://localhost').pathname
      if (pathname !== DATASET_PATH) {
        next()
        return
      }

      if (request.method !== 'GET' && request.method !== 'HEAD') {
        response.statusCode = 405
        response.setHeader('Allow', 'GET, HEAD')
        response.end('Method not allowed')
        return
      }

      void readFile(absoluteDatasetPath)
        .then((content) => {
          response.statusCode = 200
          response.setHeader('Content-Type', 'application/json; charset=utf-8')
          response.setHeader('Cache-Control', 'no-store')
          response.setHeader('Content-Length', content.byteLength)
          response.end(request.method === 'HEAD' ? undefined : content)
        })
        .catch(() => {
          response.statusCode = 404
          response.end('Review dataset not found')
        })
    })
  }

  return {
    name: 'phonia-dataset-server',
    configureServer: installMiddleware,
    configurePreviewServer: installMiddleware,
  }
}
