import { fileURLToPath } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

import { datasetServerPlugin } from './vite/datasetServer.ts'
import { mediaServerPlugin } from './vite/mediaServer.ts'
import { reviewApiPlugin } from './vite/reviewApi.ts'

const defaultMediaRoot = fileURLToPath(
  new URL('../../poc/phoneme-alignment-evaluation/data/samples/', import.meta.url),
)
const defaultReviewOutput = fileURLToPath(
  new URL(
    '../../poc/phoneme-alignment-evaluation/data/reviews/review-records.jsonl',
    import.meta.url,
  ),
)
const defaultDataset = fileURLToPath(
  new URL(
    '../../poc/phoneme-alignment-evaluation/data/reviews/review-dataset.json',
    import.meta.url,
  ),
)

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [
      react(),
      datasetServerPlugin(env.PHONIA_REVIEW_DATASET || defaultDataset),
      mediaServerPlugin(env.PHONIA_REVIEW_MEDIA_ROOT || defaultMediaRoot),
      reviewApiPlugin(env.PHONIA_REVIEW_OUTPUT || defaultReviewOutput),
    ],
  }
})
