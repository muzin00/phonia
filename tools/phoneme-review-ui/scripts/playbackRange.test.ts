import assert from 'node:assert/strict'
import test from 'node:test'

import {
  candidatePlaybackRange,
  contextPlaybackRange,
} from '../src/audio/playbackRange.ts'

test('returns the candidate segment unchanged when it is inside the audio', () => {
  assert.deepEqual(
    candidatePlaybackRange({ startSec: 1.86, endSec: 1.95 }, 8.62),
    { startSec: 1.86, endSec: 1.95 },
  )
})

test('adds context padding around a candidate segment', () => {
  assert.deepEqual(
    contextPlaybackRange({ startSec: 1.86, endSec: 1.95 }, 0.1, 8.62),
    { startSec: 1.76, endSec: 2.05 },
  )
})

test('clamps context playback to the audio boundaries', () => {
  assert.deepEqual(
    contextPlaybackRange({ startSec: 0.04, endSec: 1.95 }, 0.1, 2),
    { startSec: 0, endSec: 2 },
  )
})

