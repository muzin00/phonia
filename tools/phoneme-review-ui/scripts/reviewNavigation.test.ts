import assert from 'node:assert/strict'
import test from 'node:test'

import {
  audioFileOptions,
  nextReviewIndex,
} from '../src/domain/reviewNavigation.ts'

test('advances to the next review item', () => {
  assert.equal(nextReviewIndex(3, 10), 4)
})

test('stays on the final review item', () => {
  assert.equal(nextReviewIndex(9, 10), 9)
})

test('groups review items into selectable audio files', () => {
  assert.deepEqual(
    audioFileOptions([
      { utterance: { audioUrl: '/media/voice-1.wav' } },
      { utterance: { audioUrl: '/media/voice-1.wav' } },
      { utterance: { audioUrl: '/media/voice-2.wav?revision=1' } },
    ]),
    [
      {
        audioUrl: '/media/voice-1.wav',
        label: 'voice-1.wav',
        firstItemIndex: 0,
        itemCount: 2,
      },
      {
        audioUrl: '/media/voice-2.wav?revision=1',
        label: 'voice-2.wav',
        firstItemIndex: 2,
        itemCount: 1,
      },
    ],
  )
})
