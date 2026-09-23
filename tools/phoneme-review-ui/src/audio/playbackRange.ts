export type PlaybackRange = {
  startSec: number
  endSec: number
}

export function candidatePlaybackRange(
  segment: PlaybackRange,
  durationSec: number,
): PlaybackRange {
  return clampRange(segment.startSec, segment.endSec, durationSec)
}

export function contextPlaybackRange(
  segment: PlaybackRange,
  paddingSec: number,
  durationSec: number,
): PlaybackRange {
  return clampRange(
    segment.startSec - paddingSec,
    segment.endSec + paddingSec,
    durationSec,
  )
}

function clampRange(
  startSec: number,
  endSec: number,
  durationSec: number,
): PlaybackRange {
  return {
    startSec: Math.max(0, Math.min(startSec, durationSec)),
    endSec: Math.max(0, Math.min(endSec, durationSec)),
  }
}
