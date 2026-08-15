import { useEffect, useMemo, useState } from 'react'
import type { Bench, Replay } from './types'
import { ScoreMatrix } from './components/ScoreMatrix'
import { AnswerCard } from './components/AnswerCard'
import { BenchCharts } from './components/BenchCharts'
import { LivePanel } from './components/LivePanel'

const BUCKET_ORDER = ['regression', 'dpo_fix', 'mixed', 'all_wrong', 'all_correct']
const THEME_KEY = 'medforge.theme'
type Theme = 'system' | 'light' | 'dark'

const SET_SHORT: Record<string, string> = { cmexam: 'CMExam', 'cmb-val': 'CMB-val', medxpertqa: 'MedXpertQA' }

export default function App() {
  const [data, setData] = useState<Replay | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [setFilter, setSetFilter] = useState('all')
  const [bucketFilter, setBucketFilter] = useState('all')
  const [idx, setIdx] = useState(0)
  const [benches, setBenches] = useState<Bench[] | null>(null)
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem(THEME_KEY) as Theme) || 'system')

  // 三态主题:system 时移除属性交回 CSS 媒体查询,显式选择时钉住
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    fetch('replay.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setErr(String(e)))
    // 压测数据是可选的:还没跑过部署时,这一节自动隐藏而不是报错
    fetch('bench.json')
      .then((r) => (r.ok ? r.json() : null))
      .then((b) => b && setBenches(b))
      .catch(() => undefined)
  }, [])

  const buckets = useMemo(() => {
    if (!data) return []
    const seen = new Map<string, string>()
    data.questions.forEach((q) => seen.set(q.bucket, q.bucketLabel))
    return BUCKET_ORDER.filter((b) => seen.has(b)).map((b) => ({ key: b, label: seen.get(b)! }))
  }, [data])

  const list = useMemo(() => {
    if (!data) return []
    return data.questions.filter(
      (q) => (setFilter === 'all' || q.set === setFilter) && (bucketFilter === 'all' || q.bucket === bucketFilter),
    )
  }, [data, setFilter, bucketFilter])

  useEffect(() => setIdx(0), [setFilter, bucketFilter])

  const nextTheme = () => setTheme(theme === 'system' ? 'light' : theme === 'light' ? 'dark' : 'system')
  const themeLabel = theme === 'system' ? '跟随系统' : theme === 'light' ? '浅色' : '深色'

  const topbar = (
    <div className="topbar">
      <div className="topbar-inner">
        <div className="brand"><b>MedForge</b><span>实验台</span></div>
        <nav className="topnav">
          <a href="#finding">核心发现</a>
          <a href="#scores">成绩对照</a>
          <a href="#replay">逐题对照台</a>
          {benches && <a href="#bench">部署压测</a>}
          <a href="#live">现场提问</a>
        </nav>
        <div className="topbar-right">
          <button className="theme-btn" onClick={nextTheme} title="切换明暗">{themeLabel}</button>
          <a href="https://huggingface.co/fang04/medforge-qwen3.5-4b-dpo" target="_blank" rel="noreferrer">模型权重</a>
          <a className="cta" href="https://github.com/yiongq/medforge" target="_blank" rel="noreferrer">代码与报告 ↗</a>
        </div>
      </div>
    </div>
  )

  if (err) return <>{topbar}<div className="shell"><p className="empty">数据加载失败:{err}</p></div></>
  if (!data) return <>{topbar}<div className="shell"><p className="empty">加载中…</p></div></>

  const q = list[idx]
  const maxChars = q ? Math.max(...Object.values(q.answers).map((a) => a.chars), 1) : 1
  const cell = (run: string, set: string) => data.summary.find((s) => s.run === run && s.set === set)
  const baseMain = cell('base-v2', 'cmexam')
  const sftOldMain = cell('sft-v2', 'cmexam')
  const oldDelta = baseMain && sftOldMain ? sftOldMain.acc - baseMain.acc : null
  const peak = (l: string) => Math.max(...(benches?.find((b) => b.label === l)?.levels.map((lv) => lv.output_tok_s ?? 0) ?? [0]))
  const fp8Gain = benches && peak('bf16') ? Math.round(((peak('fp8') - peak('bf16')) / peak('bf16')) * 100) : null

  return (
    <>
      {topbar}
      <div className="shell">
        <section id="finding" className="hero" style={{ paddingTop: '3rem' }}>
          <div>
            <div className="eyebrow">W2 后训练消融 · 核心发现</div>
            <h1>对会思考的基座,<br />抄蒸馏教材是全线降分。</h1>
            <p className="lede">
              换更强的老师能大幅止损(主力卷 −11.5pp → −3.3pp),但困难卷两代教材同伤——
              说明破坏原生深推理的是「抄外部笔记」这个动作本身,与教材质量无关。
              验证器驱动的 DPO 三卷与基座持平,逐题却翻转了 413 题。
            </p>
            <div className="hero-actions">
              <a className="btn-solid" href="#scores">看四方案成绩板</a>
              <a className="btn-ghost" href="#replay">逐题翻查答卷</a>
            </div>
            <div className="stats">
              <div className="stat">
                <div className="v num">{baseMain?.acc.toFixed(1)}%</div>
                <div className="k">主力卷最高分<br />仍是未经训练的基座</div>
              </div>
              <div className="stat">
                <div className="v num down">{oldDelta?.toFixed(1)}pp</div>
                <div className="k">抄 2024 蒸馏教材<br />主力卷跌幅</div>
              </div>
              <div className="stat">
                <div className="v num">413<span style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>/2000</span></div>
                <div className="k">DPO 翻转题数<br />修 206 / 坏 207,净零</div>
              </div>
              {fp8Gain !== null && (
                <div className="stat">
                  <div className="v num up">+{fp8Gain}%</div>
                  <div className="k">FP8 峰值吞吐增益<br />且延迟更低</div>
                </div>
              )}
            </div>
          </div>
          <div className="protocol">
            <h3>评测协议 v2</h3>
            <div className="sub">固定种子抽样卷 · temperature 0 · 8192 tokens · 弃权计错</div>
            <div className="rows">
              {['cmexam', 'cmb-val', 'medxpertqa'].map((s) => {
                const r = cell('base-v2', s)
                return (
                  <div key={s}>
                    <span>{SET_SHORT[s]}</span>
                    <span className="num">{r ? `n = ${r.n.toLocaleString()}` : '—'}</span>
                  </div>
                )
              })}
              <div><span>判分</span><span className="num">规则 + LLM 兜底</span></div>
              <div><span>验证器校准</span><span className="num">200 题 · 96.5%</span></div>
              <div><span>去污染</span><span className="num">10-gram 字面查重</span></div>
            </div>
            <p className="note" style={{ marginTop: '0.9rem', marginBottom: 0 }}>
              CMB 官方 test 集不公开答案(防刷榜),故只用其 val 子集作辅助卷;主力卷为 CMExam。
            </p>
          </div>
        </section>

        <section id="scores">
          <div className="section-head">
            <h2>成绩对照</h2>
            <span className="note">四方案 × 三张考卷,同一协议下并排</span>
          </div>
          <p className="sub-note">
            条内竖线为 Wilson 95% 置信区间;右侧为相对原装基座的涨跌,落在区间内即标注「持平」。条形按 0–100 绝对刻度,不按列归一化。
          </p>
          <ScoreMatrix summary={data.summary} runs={data.meta.runs} />
        </section>

        <section id="replay">
          <div className="section-head">
            <h2>逐题对照台</h2>
            <span className="note">先看结论,想看推理再展开——字数条的长短差异本身就是现象</span>
          </div>

          <div className="controls">
            <div className="chips">
              <span className="chip-label">考卷</span>
              <button className="chip" aria-pressed={setFilter === 'all'} onClick={() => setSetFilter('all')}>全部</button>
              {Object.keys(data.meta.sets).map((k) => (
                <button key={k} className="chip" aria-pressed={setFilter === k} onClick={() => setSetFilter(k)}>
                  {SET_SHORT[k] ?? k}
                </button>
              ))}
            </div>
            <div className="chips">
              <span className="chip-label">现象</span>
              <button className="chip" aria-pressed={bucketFilter === 'all'} onClick={() => setBucketFilter('all')}>全部</button>
              {buckets.map((b) => (
                <button key={b.key} className="chip" aria-pressed={bucketFilter === b.key} onClick={() => setBucketFilter(b.key)}>
                  {b.label}
                </button>
              ))}
            </div>
            <div className="pager">
              <button onClick={() => setIdx((i) => Math.max(0, i - 1))} disabled={idx === 0}>← 上一题</button>
              <span className="count num">{list.length ? idx + 1 : 0} / {list.length}</span>
              <button onClick={() => setIdx((i) => Math.min(list.length - 1, i + 1))} disabled={idx >= list.length - 1}>下一题 →</button>
            </div>
          </div>

          {!q ? (
            <p className="empty">当前筛选下没有题目</p>
          ) : (
            <>
              <div className="stem">
                <div className="stem-meta">
                  <span className="mono">{q.id}</span>
                  <span>{SET_SHORT[q.set] ?? q.set}</span>
                  <span>{q.bucketLabel}</span>
                  {q.meta?.exam_subject && <span>{q.meta.exam_subject}</span>}
                  {q.meta?.medical_task && <span>{q.meta.medical_task}</span>}
                  <span className="gold-badge">标准答案 {q.gold}</span>
                </div>
                <div className="qtext">{q.question}</div>
                {q.options && (
                  <ol className="opts">
                    {Object.entries(q.options).map(([k, v]) => (
                      <li key={k} className={q.gold.includes(k) ? 'gold' : ''}>
                        <span className="k mono">{k}</span>
                        <span>{v}</span>
                      </li>
                    ))}
                  </ol>
                )}
              </div>

              <div className="grid">
                {data.meta.runs.map((run) => (
                  <AnswerCard key={run.key} run={run} answer={q.answers[run.key]} maxChars={maxChars} />
                ))}
              </div>
            </>
          )}
        </section>

        {benches && benches.length > 0 && (
          <section id="bench">
            <div className="section-head">
              <h2>部署压测</h2>
              <span className="note">同一份权重的 BF16 与 FP8 两档 · RTX 5090 · vLLM · 七档并发全程零失败</span>
            </div>
            <BenchCharts benches={benches} />
          </section>
        )}

        <section id="live">
          <div className="section-head">
            <h2>现场提问</h2>
            <span className="note">连到一台跑着 vLLM 的机器,用你自己的题考它</span>
          </div>
          <LivePanel />
        </section>

        <footer>
          <div className="links">
            <a href="https://github.com/yiongq/medforge" target="_blank" rel="noreferrer">代码与报告</a>
            <a href="https://huggingface.co/fang04/medforge-qwen3.5-4b-dpo" target="_blank" rel="noreferrer">模型权重</a>
            <a href="https://huggingface.co/datasets/fang04/medforge-artifacts" target="_blank" rel="noreferrer">实验产物</a>
          </div>
          <div>
            训练数据对全部考卷做过字面去污染(报告见仓库);判分由规则层 + LLM 兜底的验证器完成,
            上岗前经 200 题校准(一致率 96.5%)。
          </div>
          <div>⚠️ 研究与工程实践,输出不构成任何医疗建议。</div>
        </footer>
      </div>
    </>
  )
}
