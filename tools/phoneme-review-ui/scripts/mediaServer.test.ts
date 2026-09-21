import assert from 'node:assert/strict'
import test from 'node:test'

import { parseByteRange, resolveMediaFile } from '../vite/mediaServer.ts'

test('parses bounded and open-ended byte ranges', () => {
  assert.deepEqual(parseByteRange('bytes=10-19', 100), { start: 10, end: 19 })
  assert.deepEqual(parseByteRange('bytes=90-', 100), { start: 90, end: 99 })
  assert.deepEqual(parseByteRange('bytes=-10', 100), { start: 90, end: 99 })
})

test('rejects byte ranges outside the file', () => {
  assert.equal(parseByteRange('bytes=100-', 100), null)
  assert.equal(parseByteRange('bytes=20-10', 100), null)
})

test('resolves supported media inside the configured root', () => {
  assert.equal(
    resolveMediaFile('/data/audio', '/media/session/sample.wav'),
    '/data/audio/session/sample.wav',
  )
})

test('rejects traversal outside the configured root', () => {
  assert.equal(
    resolveMediaFile('/data/audio', '/media/%2E%2E/private.wav'),
    null,
  )
})

test('rejects unsupported file extensions', () => {
  assert.equal(resolveMediaFile('/data/audio', '/media/sample.json'), null)
})
