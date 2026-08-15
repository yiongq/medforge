import { useRef, useState } from 'react'
import { fmtChars } from '../lib'

/** live 模式:连到一台正在跑的 vLLM,现场问自己的题。
 *
 *  与回放模式的分工:回放做「四方案对照」(数据已固化、永久在线),
 *  live 做「你自己出题」(需要 GPU 在线,端点由访客自填)。
 *  端点存 localStorage——没有后端可存,也不该把别人的地址硬编码进产物。 */
const LS_KEY = 'medforge.endpoint'
const THINK_END = '</think>'

export function LivePanel() {
  const [endpoint, setEndpoint] = useState(() => localStorage.getItem(LS_KEY) ?? '')
  const [model, setModel] = useState('target')
  const [q, setQ] = useState('患者男性 65 岁,突发胸骨后压榨性疼痛 3 小时,心电图 V1-V4 导联 ST 段抬高。最可能的诊断是什么?')
  const [text, setText] = useState('')
  const [state, setState] = useState<'idle' | 'streaming' | 'error'>('idle')
  const [err, setErr] = useState('')
  const [ttft, setTtft] = useState<number | null>(null)
  const [showThinking, setShowThinking] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const idx = text.lastIndexOf(THINK_END)
  const thinking = idx === -1 ? text : text.slice(0, idx)
  const conclusion = idx === -1 ? '' : text.slice(idx + THINK_END.length)

  async function ask() {
    if (!endpoint.trim()) return setErr('先填服务地址')
    localStorage.setItem(LS_KEY, endpoint.trim())
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setText(''); setErr(''); setTtft(null); setState('streaming'); setShowThinking(true)
    const t0 = performance.now()
    let first = true
    try {
      const res = await fetch(`${endpoint.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer EMPTY' },
        body: JSON.stringify({ model, messages: [{ role: 'user', content: q }], stream: true, max_tokens: 8192, temperature: 0 }),
        signal: ac.signal,
      })
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const payload = line.slice(5).trim()
          if (payload === '[DONE]') continue
          try {
            const delta = JSON.parse(payload).choices?.[0]?.delta?.content
            if (delta) {
              if (first) { setTtft(Math.round(performance.now() - t0)); first = false }
              setText((t) => t + delta)
            }
          } catch { /* SSE 分片未闭合,等下一轮拼上 */ }
        }
      }
      setState('idle')
    } catch (e) {
      if ((e as Error).name === 'AbortError') return
      setErr(String(e)); setState('error')
    }
  }

  return (
    <div className="live">
      <div className="live-config">
        <label>
          服务地址
          <input
            value={endpoint} onChange={(e) => setEndpoint(e.target.value)} spellCheck={false}
            placeholder="http://<你的-vLLM-主机>:8000/v1"
          />
        </label>
        <label className="model-field">
          模型名
          <input value={model} onChange={(e) => setModel(e.target.value)} spellCheck={false} />
        </label>
      </div>
      <textarea value={q} onChange={(e) => setQ(e.target.value)} rows={3} placeholder="输入一道医学题…" />
      <div className="live-actions">
        <button className="primary" onClick={ask} disabled={state === 'streaming'}>
          {state === 'streaming' ? '作答中…' : '提问'}
        </button>
        {state === 'streaming' && <button onClick={() => abortRef.current?.abort()}>停止</button>}
        {ttft !== null && <span className="num live-metric">首字 {ttft} ms</span>}
        {text && <span className="num live-metric">{fmtChars(text.length)}</span>}
        {err && <span className="live-err">{err}</span>}
      </div>

      {!endpoint && !text && (
        <p className="note">
          回放模式的四方案对照无需任何服务;这里是「现场提问」——需要一台跑着 vLLM 的机器。
          起服务:<code>vllm serve fang04/medforge-qwen3.5-4b-dpo --served-model-name target --port 8000</code>,
          把它的地址填上面即可(浏览器直连,vLLM 默认放行跨域)。
        </p>
      )}

      {text && (
        <div className="card ok live-out">
          <div className="card-body">
            {conclusion && <div className="conclusion">{conclusion}</div>}
            {thinking && (
              <>
                <button className="thinking-toggle" onClick={() => setShowThinking(!showThinking)}>
                  {showThinking ? '收起思考过程' : `展开思考过程 · ${fmtChars(thinking.length)}`}
                </button>
                {showThinking && <div className="thinking">{thinking}</div>}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
