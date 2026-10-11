import {wav16, base64} from './audio.js';
const $ = (id) => document.getElementById(id);
let token, episode, capture, busy = false;
const status = (message, error = false) => { $('status').textContent = message; $('status').classList.toggle('error', error); };
async function api(path, value) {
  const response = await fetch(path, value === undefined ? {} : {
    method: 'POST', headers: {'Content-Type':'application/json', 'X-Phonia-Token':token}, body:JSON.stringify(value)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error);
  return result;
}
function controls() {
  const active = episode?.status === 'collecting';
  $('record').disabled = !active || busy || Boolean(capture);
  $('stop').disabled = !capture || capture.finishing;
  $('file').disabled = !active || busy || Boolean(capture);
  $('abort').disabled = !active || busy || Boolean(capture);
  $('next').hidden = active;
}
function render() {
  $('setup').hidden = true; $('recording').hidden = false;
  $('episode-title').textContent = {practice:'練習',enrollment:'登録用の録音',query:'照合用の録音'}[episode.role];
  $('metadata').textContent = `${episode.participant} / ${episode.device} / ${episode.session} / ${episode.recorded_on}`;
  $('counter').textContent = `${episode.clips.length} / ${episode.maximum_clips} 回 · 合計 ${episode.seconds.toFixed(1)} 秒`;
  $('prompt').textContent = episode.prompt.text;
  const active = episode.status === 'collecting';
  $('prompt').hidden = !active;
  $('guidance').textContent = active ? `1回の録音は最大${episode.remaining_clip_seconds.toFixed(1)}秒。${episode.role === 'enrollment' ? '20回で登録用の声を集めます。' : '母音がそろえば終了します。不足があれば追加の文を案内します。'}` :
    episode.status === 'complete' ? '録音が完了しました。本人認証の判定ではありません。' : episode.status === 'aborted' ? 'この試行を終了しました。記録は保存されています。' : '収集上限に達したか、登録処理に失敗しました。不足や失敗を含めて記録を保存しました。';
  $('counts').replaceChildren(...['a','i','u','e','o'].map((v) => {
    const node = document.createElement('div'); node.className = `vowel ${episode.missing_vowels.includes(v) ? 'missing' : ''}`;
    const label = document.createElement('span'); label.textContent = v;
    const count = document.createElement('strong'); count.textContent = episode.counts[v];
    node.append(label, count); return node;
  }));
  $('history').replaceChildren(...episode.clips.map((clip, i) => {
    const p = document.createElement('p');
    p.textContent = `${i + 1}回目 · ${clip.seconds.toFixed(1)}秒 · ${{processed:'抽出済み',failed:'処理失敗',unusable:'有効な音素なし',interrupted:'処理中断',processing:'処理中'}[clip.status]}${clip.error ? `：${clip.error}` : ''}`;
    return p;
  }));
  sessionStorage.setItem('phonia-episode', episode.id); controls();
}
async function submit(buffer, origin, settings = {}) {
  busy = true; controls(); status('音声を保存し、母音を抽出しています…');
  try {
    episode = await api('/api/clip', {episode_id:episode.id,prompt_id:episode.prompt.id,wav_base64:base64(buffer),origin,audio_settings:settings});
    render(); status(episode.status === 'collecting' ? '保存しました。次の文を録音してください。' : '試行の記録を保存しました。');
  } catch (error) {
    // The server may have persisted this attempt even if the response was lost.
    try { episode = await api(`/api/episode?id=${episode.id}`); render(); } catch { /* keep prior state and report */ }
    status(error.message, true);
  } finally {busy = false; controls();}
}
async function release(item) {
  clearInterval(item.timer);
  clearTimeout(item.deadline);
  item.stream?.getTracks().forEach((track) => track.stop());
  item.node?.disconnect(); item.source?.disconnect();
  if (item.context && item.context.state !== 'closed') await item.context.close();
}
async function finish(item, pcm) {
  if (item.finishing) return;
  item.finishing = true; busy = true; controls();
  try {
    await release(item); capture = null;
    $('timer').textContent = `${(pcm.length / item.rate).toFixed(1)}秒`;
    await submit(wav16(pcm,item.rate),'microphone',item.settings);
  } catch(error) {status(error.message,true);}
  finally {capture = null; busy = false; controls();}
}
$('record').onclick = async () => {
  busy = true; controls(); status('マイクの準備中です…');
  const item = {}; capture = item;
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) throw new Error('対応ブラウザーで http://127.0.0.1:8765 を開いてください。');
    item.stream = await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    item.context = new AudioContext(); item.rate = item.context.sampleRate;
    await item.context.audioWorklet.addModule('/capture.js');
    item.settings = item.stream.getAudioTracks()[0].getSettings();
    // Device identifiers are unnecessary; retain only signal-processing settings.
    item.settings = Object.fromEntries(['sampleRate','channelCount','echoCancellation','noiseSuppression','autoGainControl','latency'].filter(k => k in item.settings).map(k=>[k,item.settings[k]]));
    item.node = new AudioWorkletNode(item.context,'capture',{processorOptions:{maximumFrames:Math.round(episode.remaining_clip_seconds * item.rate)}});
    item.node.port.onmessage = ({data}) => void finish(item,data);
    item.source = item.context.createMediaStreamSource(item.stream);
    item.source.connect(item.node); item.node.connect(item.context.destination);
    await item.context.resume();
    item.deadline = setTimeout(() => item.node.port.postMessage('stop'), episode.remaining_clip_seconds * 1000 + 200);
    const started = performance.now();
    item.timer = setInterval(() => {$('timer').textContent = `${((performance.now()-started)/1000).toFixed(1)}秒`;},100);
    busy = false; controls(); status('録音中です。表示文を読んでください。');
  } catch(error) {await release(item); capture = null; busy = false; controls(); status(error.message,true);}
};
$('stop').onclick = () => {if (capture?.node) {capture.node.port.postMessage('stop'); $('stop').disabled = true;}};
$('file').onchange = async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 2_000_000) {status('ファイルが大きすぎます。最大8秒のモノラル16bit WAVを使用してください。',true); return;}
  await submit(await file.arrayBuffer(),'file'); event.target.value = '';
};
$('setup-form').onsubmit = async (event) => {
  event.preventDefault(); $('begin').disabled = true;
  try {
    const fields = Object.fromEntries(new FormData(event.target));
    episode = await api('/api/episode',{...fields,consent:fields.consent === 'on'}); render(); status('準備できました。録音開始ボタンを押してください。');
  } catch(error) {status(error.message,true);} finally {$('begin').disabled = false;}
};
$('abort').onclick = async () => {
  busy = true; controls();
  try {episode = await api('/api/abort',{episode_id:episode.id});render();status('試行を終了し、記録を保存しました。');}
  catch(error) {status(error.message,true);} finally {busy = false;controls();}
};
$('next').onclick = () => {sessionStorage.removeItem('phonia-episode');episode = null;$('setup').hidden = false;$('recording').hidden = true;status('次の試行の情報を入力してください。');};
window.addEventListener('pagehide', () => {if (capture) {capture.stream?.getTracks().forEach(t=>t.stop());void release(capture);}});
const date = new Date(); date.setMinutes(date.getMinutes() - date.getTimezoneOffset());
document.querySelector('[name=recorded_on]').value = date.toISOString().slice(0,10);
try {
  const info = await api('/api/info'); token = info.token; $('begin').disabled = false;
  status('接続しました。まず練習で録音の流れを確認できます。');
  const previous = sessionStorage.getItem('phonia-episode');
  if (previous) {try {episode = await api(`/api/episode?id=${previous}`);render();} catch {sessionStorage.removeItem('phonia-episode');}}
} catch(error) {status(`接続できません：${error.message}`,true);}
