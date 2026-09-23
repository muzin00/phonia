export function nextReviewIndex(currentIndex: number, itemCount: number): number {
  if (itemCount <= 0) {
    return 0
  }
  return Math.min(itemCount - 1, currentIndex + 1)
}

export type AudioFileOption = {
  audioUrl: string
  label: string
  firstItemIndex: number
  itemCount: number
}

export function audioFileOptions(
  items: Array<{
    utterance: { audioUrl: string }
  }>,
): AudioFileOption[] {
  const options = new Map<string, AudioFileOption>()
  items.forEach((item, index) => {
    const audioUrl = item.utterance.audioUrl
    const current = options.get(audioUrl)
    if (current !== undefined) {
      current.itemCount += 1
      return
    }
    options.set(audioUrl, {
      audioUrl,
      label: audioFileName(audioUrl),
      firstItemIndex: index,
      itemCount: 1,
    })
  })
  return [...options.values()]
}

function audioFileName(audioUrl: string): string {
  const pathname = audioUrl.split(/[?#]/, 1)[0] ?? audioUrl
  const name = pathname.split('/').filter(Boolean).at(-1)
  return name === undefined ? audioUrl : decodeURIComponent(name)
}
