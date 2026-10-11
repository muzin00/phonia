/* Capture bounded mono PCM without ScriptProcessor or recording compression. */
class Capture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = new Float32Array(options.processorOptions.maximumFrames);
    this.offset = 0;
    this.finished = false;
    this.port.onmessage = ({data}) => { if (data === 'stop') this.finish(); };
  }
  finish() {
    if (this.finished) return;
    this.finished = true;
    const pcm = this.buffer.slice(0, this.offset);
    this.port.postMessage(pcm, [pcm.buffer]);
  }
  process(inputs) {
    if (this.finished) return false;
    const input = inputs[0]?.[0];
    if (input) {
      const count = Math.min(input.length, this.buffer.length - this.offset);
      this.buffer.set(input.subarray(0, count), this.offset);
      this.offset += count;
      if (this.offset === this.buffer.length) this.finish();
    }
    return !this.finished;
  }
}
registerProcessor('capture', Capture);
