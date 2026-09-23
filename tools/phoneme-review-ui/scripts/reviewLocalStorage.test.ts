import assert from 'node:assert/strict'
import test from 'node:test'

import {
  loadLocalReviewState,
  localReviewStorageKey,
  saveLocalReviewState,
} from '../src/domain/reviewLocalStorage.ts'

class MemoryStorage {
  readonly values = new Map<string, string>()

  getItem(key: string): string | null {
    return this.values.get(key) ?? null
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value)
  }
}

test('persists and restores dataset-scoped review drafts', () => {
  const storage = new MemoryStorage()
  const state = {
    currentIndex: 12,
    drafts: {
      item1: {
        candidateAnswers: { A: { audible: 'yes' } },
        reviewStatus: 'accepted' as const,
      },
    },
  }

  saveLocalReviewState(storage, 'dataset', '3', state)

  assert.deepEqual(loadLocalReviewState(storage, 'dataset', '3'), state)
  assert.equal(loadLocalReviewState(storage, 'dataset', '2'), null)
  assert.ok(storage.values.has(localReviewStorageKey('dataset', '3')))
})

test('ignores invalid local review state', () => {
  const storage = new MemoryStorage()
  storage.setItem(localReviewStorageKey('dataset', '3'), '{invalid')

  assert.equal(loadLocalReviewState(storage, 'dataset', '3'), null)
})
