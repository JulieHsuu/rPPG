const TARGET_SECONDS = 50

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
const median = values => {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}
const mean = values => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length)
const std = values => {
  const m = mean(values)
  return Math.sqrt(mean(values.map(v => (v - m) ** 2)))
}
const movingAverage = (x, n) => x.map((_, i) => {
  const a = Math.max(0, i - n + 1)
  return mean(x.slice(a, i + 1))
})
const detrend = (x, n) => {
  const trend = movingAverage(x, n)
  return x.map((v, i) => v - trend[i])
}

function spectrumRate(signal, fps, low, high) {
  if (signal.length < fps * 8) return [null, 0]
  const x = signal.slice(-Math.min(signal.length, Math.round(fps * TARGET_SECONDS)))
  const center = mean(x)
  const powers = []
  const step = 1 / (x.length / fps * 5)
  for (let f = low; f <= high; f += step) {
    let re = 0; let im = 0
    for (let i = 0; i < x.length; i++) {
      const window = .5 - .5 * Math.cos(2 * Math.PI * i / Math.max(1, x.length - 1))
      const phase = 2 * Math.PI * f * i / fps
      re += (x[i] - center) * window * Math.cos(phase)
      im -= (x[i] - center) * window * Math.sin(phase)
    }
    powers.push([f, re * re + im * im])
  }
  if (!powers.length) return [null, 0]
  const peak = powers.reduce((a, b) => b[1] > a[1] ? b : a)
  const floor = median(powers.map(p => p[1])) || 1e-12
  return [peak[0] * 60, clamp((peak[1] / floor - 1) / 14, 0, 1)]
}

function pulseSignal(samples, engine) {
  const recent = samples.slice(-1250)
  const mr = mean(recent.map(s => s.r)); const mg = mean(recent.map(s => s.g)); const mb = mean(recent.map(s => s.b))
  const r = recent.map(s => s.r / mr - 1); const g = recent.map(s => s.g / mg - 1); const b = recent.map(s => s.b / mb - 1)
  let raw
  if (engine === 'chrom') {
    const x = r.map((v, i) => 3 * v - 2 * g[i])
    const y = r.map((v, i) => 1.5 * v + g[i] - 1.5 * b[i])
    const alpha = std(x) / Math.max(1e-9, std(y))
    raw = x.map((v, i) => v - alpha * y[i])
  } else {
    const x = g.map((v, i) => v - b[i])
    const y = g.map((v, i) => v + b[i] - 2 * r[i])
    const alpha = std(x) / Math.max(1e-9, std(y))
    raw = x.map((v, i) => v + alpha * y[i])
  }
  return detrend(raw, Math.max(3, Math.round(recent.length / Math.max(8, recent.at(-1).t - recent[0].t) * 1.2)))
}

function estimateHrv(pulse, fps, hr) {
  if (!hr || pulse.length < fps * 35) return null
  const smooth = movingAverage(pulse, Math.max(2, Math.round(fps * .08)))
  const threshold = std(smooth) * .12
  const minGap = Math.round(fps * 60 / hr * .58)
  const peaks = []
  for (let i = 1; i < smooth.length - 1; i++) {
    if (smooth[i] > smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > threshold && (!peaks.length || i - peaks.at(-1) >= minGap)) peaks.push(i)
  }
  let ibi = peaks.slice(1).map((p, i) => (p - peaks[i]) / fps * 1000).filter(v => v >= 333 && v <= 1500)
  if (ibi.length < 10) return null
  const expected = 60000 / hr
  ibi = ibi.filter(v => Math.abs(v - expected) < expected * .25)
  if (ibi.length < 10) return null
  const diffs = ibi.slice(1).map((v, i) => v - ibi[i]).filter(v => Math.abs(v) < 180)
  if (diffs.length < 8) return null
  const raw = Math.sqrt(mean(diffs.map(v => v * v)))
  return Math.round(clamp(15 + .25 * (raw - 15), 1, 120) * 10) / 10
}

function wellness(hr, hrv, q, motion, exposure) {
  if (hr == null || hrv == null || q < .42) return [null, null]
  const h = clamp((hrv - 15) / 45, 0, 1); const rest = clamp(1 - Math.abs(hr - 70) / 45, 0, 1)
  const still = clamp(1 - motion, 0, 1); const light = clamp(exposure, 0, 1)
  const evidence = 100 * (1 - (.55 * h + .45 * rest)); const reliability = clamp((q - .42) / .38, 0, 1)
  const stress = Math.round(clamp(50 + (evidence - 50) * reliability, 15, 85) * 10) / 10
  const five = v => Math.round(clamp(1 + 4 * v, 1, 5) * 10) / 10
  return [stress, {activity:five(.5*(1-rest)+.5*q),sleep:five(.55*h+.45*q),equilibrium:five(.4*h+.35*still+.25*q),metabolism:five(.45*rest+.3*q+.25*light),health:five(.4*h+.35*rest+.25*q),relaxation:five(.6*h+.4*still)}]
}

function bloodPressureProxy(hr, hrv, stress, q) {
  if (hr == null || hrv == null || stress == null || q < .42) return [null, null]
  const hrvLoad = clamp(45 - hrv, -25, 35); const stressLoad = clamp(stress - 50, -30, 30)
  const systolic = Math.round(clamp(107 + .06 * hr + .035 * hrvLoad + .025 * stressLoad, 95, 145))
  const diastolic = Math.round(clamp(73 + .075 * hr + .025 * hrvLoad + .015 * stressLoad, 60, Math.min(95, systolic - 20)))
  return [systolic, diastolic]
}

const FIXED_FACE_BOX = {x:.32,y:.12,w:.36,h:.68}

// Restore the pre-regression sample locations and averaging. All three
// regions are inside FIXED_FACE_BOX; no frame-dependent pixel mask is applied.
function sampleRois(ctx, w, h) {
  const rois = [[.42,.21,.16,.12],[.35,.43,.12,.13],[.53,.43,.12,.13]]
  const totals = []
  for (const [rx, ry, rw, rh] of rois) {
    const data = ctx.getImageData(Math.floor(w*rx),Math.floor(h*ry),Math.max(1,Math.floor(w*rw)),Math.max(1,Math.floor(h*rh))).data
    let r=0,g=0,b=0,n=0
    for(let i=0;i<data.length;i+=16){r+=data[i];g+=data[i+1];b+=data[i+2];n++}
    totals.push({r:r/n,g:g/n,b:b/n})
  }
  return {r:median(totals.map(v=>v.r)),g:median(totals.map(v=>v.g)),b:median(totals.map(v=>v.b))}
}

export function createLocalRppg(onVitals) {
  let samples=[]; let last=null; let validSeconds=0; let completed=false; let previous=null; let hrHistory=[]; let rrHistory=[]; let hrvHistory=[]; let lastEmit=0
  return {
    reset(){samples=[];last=null;validSeconds=0;completed=false;previous=null;hrHistory=[];rrHistory=[];hrvHistory=[];lastEmit=0},
    process(ctx,w,h,t,engine){
      if(completed)return
      const trackedBox=FIXED_FACE_BOX
      const rgb=sampleRois(ctx,w,h)
      const brightness=.299*rgb.r+.587*rgb.g+.114*rgb.b
      const exposure=clamp(1-Math.abs(brightness-125)/115,0,1)
      const motion=previous?clamp((Math.abs(rgb.r-previous.r)+Math.abs(rgb.g-previous.g)+Math.abs(rgb.b-previous.b))/12,0,1):0
      previous=rgb
      const valid=exposure>.32&&motion<.72
      if(valid){if(last!=null&&t-last<.25)validSeconds+=t-last;last=t;samples.push({...rgb,t});while(samples.length&&t-samples[0].t>TARGET_SECONDS)samples.shift()}
      if(t-lastEmit<.24)return;lastEmit=t
      const analysisSpan=samples.length>1?samples.at(-1).t-samples[0].t:0; const fps=analysisSpan>0?(samples.length-1)/analysisSpan:0
      let hr=null,rr=null,hrv=null,spo2=null,wave=[],spectral=0
      // A short window often locks onto a sub-harmonic (for example 42 instead
      // of 84 bpm). Keep collecting visibly, but withhold HR until 18 seconds.
      if(!(fps>10&&samples.length>fps*18)){
        const startupQuality=clamp(exposure*(1-.45*motion),0,1)
        onVitals({type:'vitals',status:valid?'measuring':validSeconds>0?'paused':'low_quality',hr:null,rr:null,hrv_rmssd:null,spo2:null,stress_index:null,wellness:null,bp_systolic:null,bp_diastolic:null,signal_quality:Math.round(startupQuality*1000)/1000,waveform:[],bbox:trackedBox,engine,fps:fps?Math.round(fps*10)/10:null,window_seconds:Math.round(validSeconds*10)/10,measurement_progress:Math.round(Math.min(100,validSeconds*2)*10)/10,motion:Math.round(motion*1000)/1000,exposure:Math.round(exposure*1000)/1000,roi_quality:Math.round(startupQuality*1000)/1000})
        return
      }
      if(fps>10&&samples.length>fps*18){
        const pulse=pulseSignal(samples,engine); [hr,spectral]=spectrumRate(pulse,fps,.7,3)
        const q=clamp(spectral*(.72+.28*(1-motion))*exposure,0,1)
        if(q<.4)hr=null; if(hr){hrHistory.push(hr);hrHistory=hrHistory.slice(-9);hr=median(hrHistory)}
        if(validSeconds>=30){const intensity=detrend(samples.map(s=>(s.r+s.g+s.b)/3),Math.round(fps*4));[rr]=spectrumRate(intensity,fps,.1,.5);if(rr){rrHistory.push(rr);rrHistory=rrHistory.slice(-9);rr=median(rrHistory)}hrv=estimateHrv(pulse,fps,hr);if(hrv){hrvHistory.push(hrv);hrvHistory=hrvHistory.slice(-9);hrv=median(hrvHistory)}const mr=mean(samples.map(s=>s.r)),mg=mean(samples.map(s=>s.g));const red=std(detrend(samples.map(s=>s.r/mr),Math.round(fps))),green=std(detrend(samples.map(s=>s.g/mg),Math.round(fps)));spo2=clamp(100-5*red/Math.max(green,1e-9),90,99)}
        wave=pulse.slice(-Math.round(fps*8));const scale=Math.max(...wave.map(Math.abs),1e-9);wave=wave.filter((_,i)=>i%Math.max(1,Math.floor(wave.length/180))===0).map(v=>Math.round(v/scale*10000)/10000)
        const [stress,well]=wellness(hr,hrv,q,motion,exposure);const [bpSystolic,bpDiastolic]=bloodPressureProxy(hr,hrv,stress,q);const complete=validSeconds>=TARGET_SECONDS&&hr!=null
        if(complete)completed=true
        onVitals({type:'vitals',status:complete?'complete':!valid?'paused':validSeconds>=30?(q>=.45?'good':'low_quality'):'measuring',hr:hr?Math.round(hr*10)/10:null,rr:rr?Math.round(clamp(rr*2,8,40)*10)/10:null,hrv_rmssd:hrv?Math.round(hrv*5)/10:null,spo2:spo2?Math.round(spo2*10)/10:null,spo2_experimental:true,stress_index:stress,wellness:well,bp_systolic:bpSystolic,bp_diastolic:bpDiastolic,bp_experimental:true,signal_quality:Math.round(q*1000)/1000,waveform:wave,bbox:trackedBox,engine,fps:Math.round(fps*10)/10,window_seconds:Math.round(validSeconds*10)/10,measurement_progress:Math.round(Math.min(100,validSeconds*2)*10)/10,motion:Math.round(motion*1000)/1000,exposure:Math.round(exposure*1000)/1000,roi_quality:Math.round(q*1000)/1000})
      }
    }
  }
}
