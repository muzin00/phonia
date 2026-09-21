import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js'

import {
  candidatePlaybackRange,
  contextPlaybackRange,
  type PlaybackRange,
} from '../audio/playbackRange.ts'
import type { ReviewCandidate } from '../domain/reviewDataset.ts'

type AvailableCandidate = Extract<ReviewCandidate, { status: 'available' }>
type WaveformState =
  | { status: 'loading'; percent: number }
  | { status: 'ready'; durationSec: number }
  | { status: 'error'; message: string }

export function WaveformPanel({
  audioUrl,
  candidate,
  contextPaddingSec,
}: {
  audioUrl: string
  candidate: AvailableCandidate
  contextPaddingSec: number
}) {
  const containerRef = useRef<HTMLDivElement>(null)
  const waveSurferRef = useRef<WaveSurfer>(null)
  const [waveformState, setWaveformState] = useState<WaveformState>({
    status: 'loading',
    percent: 0,
  })
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    const container = containerRef.current
    if (container === null) {
      return
    }

    const regions = RegionsPlugin.create()
    const waveSurfer = WaveSurfer.create({
      container,
      url: audioUrl,
      height: 148,
      waveColor: '#9ed8ea',
      progressColor: '#247fa1',
      cursorColor: '#0f5d78',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      normalize: true,
      backend: 'WebAudio',
      plugins: [regions],
    })
    waveSurferRef.current = waveSurfer

    waveSurfer.on('loading', (percent) => {
      setWaveformState({ status: 'loading', percent })
    })
    waveSurfer.on('ready', (durationSec) => {
      if (
        candidate.segment.startSec >= durationSec ||
        candidate.segment.endSec > durationSec
      ) {
        setWaveformState({
          status: 'error',
          message: '候補区間が音声の長さを超えています。',
        })
        return
      }

      regions.addRegion({
        id: candidate.id,
        start: candidate.segment.startSec,
        end: candidate.segment.endSec,
        content: `候補 ${candidate.id}`,
        color: 'rgb(35 170 210 / 30%)',
        drag: false,
        resize: false,
      })
      setWaveformState({ status: 'ready', durationSec })
    })
    waveSurfer.on('play', () => setIsPlaying(true))
    waveSurfer.on('pause', () => setIsPlaying(false))
    waveSurfer.on('finish', () => setIsPlaying(false))
    waveSurfer.on('error', (error) => {
      setWaveformState({
        status: 'error',
        message: `音声を読み込めませんでした: ${error.message}`,
      })
    })

    return () => {
      waveSurferRef.current = null
      waveSurfer.destroy()
    }
  }, [audioUrl, candidate])

  const playRange = (range: PlaybackRange) => {
    const waveSurfer = waveSurferRef.current
    if (waveSurfer === null) {
      return
    }
    void waveSurfer.play(range.startSec, range.endSec).catch((error: unknown) => {
      setWaveformState({
        status: 'error',
        message: error instanceof Error ? error.message : '音声を再生できませんでした。',
      })
    })
  }

  const durationSec =
    waveformState.status === 'ready' ? waveformState.durationSec : null

  return (
    <div className="waveform-panel">
      <div className="waveform-heading">
        <div>
          <p className="waveform-label">Waveform · Candidate {candidate.id}</p>
          <p className="waveform-help">水色の範囲がモデルの推定区間です</p>
        </div>
        {durationSec !== null && (
          <span className="audio-duration">全長 {formatSeconds(durationSec)}</span>
        )}
      </div>

      <div className="waveform-canvas-wrap">
        <div ref={containerRef} className="waveform-canvas" />
        {waveformState.status === 'loading' && (
          <div className="waveform-overlay" aria-live="polite">
            波形を読み込み中… {Math.round(waveformState.percent)}%
          </div>
        )}
        {waveformState.status === 'error' && (
          <div className="waveform-overlay waveform-overlay-error" role="alert">
            {waveformState.message}
          </div>
        )}
      </div>

      <div className="playback-controls" aria-label="音声再生">
        <button
          type="button"
          className="button button-secondary"
          disabled={durationSec === null}
          onClick={() => {
            const waveSurfer = waveSurferRef.current
            if (waveSurfer === null) {
              return
            }
            if (waveSurfer.isPlaying()) {
              waveSurfer.pause()
            } else {
              void waveSurfer.play()
            }
          }}
        >
          {isPlaying ? '一時停止' : '原音声を再生'}
        </button>
        <button
          type="button"
          className="button button-secondary"
          disabled={durationSec === null}
          onClick={() => {
            if (durationSec !== null) {
              playRange(
                contextPlaybackRange(
                  candidate.segment,
                  contextPaddingSec,
                  durationSec,
                ),
              )
            }
          }}
        >
          前後{Math.round(contextPaddingSec * 1000)}msを含めて再生
        </button>
        <button
          type="button"
          className="button button-primary"
          disabled={durationSec === null}
          onClick={() => {
            if (durationSec !== null) {
              playRange(candidatePlaybackRange(candidate.segment, durationSec))
            }
          }}
        >
          候補{candidate.id}だけ再生
        </button>
      </div>
    </div>
  )
}

function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(3)} s`
}
