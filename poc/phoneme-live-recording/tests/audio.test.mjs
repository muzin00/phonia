import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const directory = new URL('../web/',import.meta.url);
const audio = await import('data:text/javascript;base64,' + Buffer.from(readFileSync(new URL('audio.js',directory))).toString('base64'));
const wav = audio.wav16(new Float32Array([-1,0,0.5,1,2]),48000);
const view = new DataView(wav);
assert.equal(view.getUint32(24,true),48000);
assert.equal(view.getUint32(40,true),10);
assert.deepEqual([...new Int16Array(wav.slice(44))],[-32768,0,16384,32767,32767]);
assert.equal(Buffer.from(audio.base64(wav),'base64').length,54);
let Capture;
const sent=[];
vm.runInNewContext(readFileSync(new URL('capture.js',directory),'utf8'),{
  AudioWorkletProcessor:class {constructor(){this.port={postMessage:x=>sent.push(x)};}},
  registerProcessor:(name,processor)=>{assert.equal(name,'capture');Capture=processor;},Float32Array
});
const worklet=new Capture({processorOptions:{maximumFrames:5}});
worklet.process([[new Float32Array([1,2,3])]]);
worklet.process([[new Float32Array([4,5,6])]]);
assert.deepEqual([...sent[0]],[1,2,3,4,5]);
worklet.port.onmessage({data:'stop'});assert.equal(sent.length,1);
const early=new Capture({processorOptions:{maximumFrames:5}});
early.process([[new Float32Array([1,2])]]);early.port.onmessage({data:'stop'});
assert.deepEqual([...sent[1]],[1,2]);
console.log('PASS WAV PCM encoding, bounded capture, early stop, and single flush');
