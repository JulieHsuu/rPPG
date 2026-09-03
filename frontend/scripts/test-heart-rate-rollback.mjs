import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'

const baseSource = execFileSync('git', ['show', '29f6637:src/lib/localRppg.js'], { encoding: 'utf8' })
const currentSource = await readFile(new URL('../src/lib/localRppg.js', import.meta.url), 'utf8')
const load = source => import(`data:text/javascript;base64,${Buffer.from(source + '\nexport { sampleRois, spectrumRate, pulseSignal };').toString('base64')}`)
const base = await load(baseSource)
const current = await load(currentSource)

// Synthetic pixels vary by position: this catches changes in ROI coordinates,
// stride and frame-dependent pixel selection, not just uniform-frame averages.
const reads = []
const spatialContext = {
  getImageData(x, y, w, h) {
    reads.push([x, y, w, h])
    const data = new Uint8ClampedArray(w*h*4)
    for (let i = 0; i < w*h; i++) {
      data[i*4] = (x + i%w)*3%256
      data[i*4+1] = (y + Math.floor(i/w))*7%256
      data[i*4+2] = (i + x + y)%256
    }
    return { data }
  },
}
assert.deepEqual(current.sampleRois(spatialContext, 640, 360), base.sampleRois(spatialContext, 640, 360))
for (const [x,y,w,h] of reads) {
  assert(x >= Math.floor(.32*640) && x+w <= Math.ceil(.68*640))
  assert(y >= Math.floor(.12*360) && y+h <= Math.ceil(.80*360))
}

for (const bpm of [48, 84, 120]) for (const engine of ['pos', 'chrom']) {
  const fps = 15
  const samples = Array.from({length: 600}, (_,i) => {
    const t = i/fps, pulse = Math.sin(2*Math.PI*bpm/60*t)
    return {t, r:150+.5*pulse, g:110+2*pulse, b:85+.15*pulse}
  })
  const actual = current.spectrumRate(current.pulseSignal(samples,engine),fps,.7,3)
  const expected = base.spectrumRate(base.pulseSignal(samples,engine),fps,.7,3)
  assert.deepEqual(actual, expected)
  assert(Math.abs(actual[0]-bpm)<2)
  console.log(`${engine} known=${bpm}, restored=${actual[0].toFixed(1)}, baseline=${expected[0].toFixed(1)}`)
}

function replay(module, constant = false) {
  const messages = []
  const analyzer = module.createLocalRppg(message => messages.push(message))
  let time = 0
  const ctx = { getImageData(_x,_y,w,h) {
    const pulse = constant ? 0 : Math.sin(2*Math.PI*1.4*time)
    const data = new Float64Array(w*h*4)
    for(let i=0;i<data.length;i+=4){data[i]=150+.5*pulse;data[i+1]=110+2*pulse;data[i+2]=85+.15*pulse;data[i+3]=255}
    return {data}
  }}
  for(let i=0;i<=15*53;i++){time=i/15;analyzer.process(ctx,160,90,time,'pos')}
  return messages
}
const restored = replay(current), baseline = replay(base)
assert(restored.some(m => m.window_seconds>0 && m.window_seconds<1))
assert(restored.filter(m => m.window_seconds<18).every(m=>m.hr===null))
assert.equal(restored.filter(m=>m.status==='complete').length,1)
assert.equal(restored.at(-1).status,'complete')
assert.equal(restored.at(-1).hr,baseline.at(-1).hr)
assert.equal(restored.at(-1).signal_quality,baseline.at(-1).signal_quality)
assert.equal(restored.at(-1).measurement_progress,100)
assert(replay(current,true).every(m=>m.hr===null && m.status!=='complete'))
console.log('PASS: baseline sampling, true 48 bpm preserved, fixed guide, immediate progress, completion lock, flat-signal rejection. Synthetic tests are not camera/clinical validation.')
