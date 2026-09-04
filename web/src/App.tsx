import { useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import type { Bench, Replay, ScoreRow } from './types'
import { ScoreMatrix } from './components/ScoreMatrix'
import { AnswerCard } from './components/AnswerCard'
import { BenchCharts } from './components/BenchCharts'
import { LivePanel } from './components/LivePanel'

const BUCKET_ORDER = ['decoding_fix', 'regression', 'dpo_fix', 'mixed', 'all_wrong', 'all_correct']
const THEME_KEY = 'medforge.theme'
type Theme = 'system' | 'light' | 'dark'

const SET_SHORT: Record<string, string> = { cmexam: 'CMExam', 'cmb-val': 'CMB-val', medxpertqa: 'MedXpertQA' }
const SET_ORDER = ['cmexam', 'cmb-val', 'medxpertqa']
const BASELINE = 'base-v2'
const V3_RUN = 'base-v3-sample'

/** 首屏一律用严格口径(写完 ∧ 有结论 ∧ 答对);没跑过 usability 的 run 才退回宽口径。 */
const strictOf = (r: ScoreRow | undefined) => (r ? r.strict ?? r.acc : null)

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

  // 首屏数字全部从 summary 现算,不硬编码:改了协议或重跑了某一臂,文案会跟着动
  const baseMain = cell(BASELINE, 'cmexam')
  const v3Main = cell(V3_RUN, 'cmexam')
  const baseStrict = strictOf(baseMain)
  const v3Strict = strictOf(v3Main)
  const decodeGain = baseStrict !== null && v3Strict !== null ? v3Strict - baseStrict : null
  const v3Rows = data.summary.filter((r) => r.run === V3_RUN && r.finished !== undefined)
  const v3Finished = v3Rows.length ? Math.min(...v3Rows.map((r) => r.finished!)) : null
  const baseFinished = data.summary
    .filter((r) => r.run === BASELINE && r.finished !== undefined)
    .map((r) => r.finished!)
  const hasV3 = v3Main !== undefined
  const peak = (l: string) => Math.max(...(benches?.find((b) => b.label === l)?.levels.map((lv) => lv.output_tok_s ?? 0) ?? [0]))
  const fp8Gain = benches && peak('bf16') ? Math.round(((peak('fp8') - peak('bf16')) / peak('bf16')) * 100) : null

  return (
    <>
      {topbar}
      <div className="shell">
        <section id="finding" className="hero" style={{ paddingTop: '3rem' }}>
          <div>
            <div className="eyebrow">P2 解码裁决 · 核心发现</div>
            <h1>W2 的成绩表,<br />量的是解码方式,不是模型。</h1>
            <p className="lede">
              {hasV3 ? (
                <>
                  同一份 Qwen3.5-4B 基座权重,不训练、不改提示词,只把贪心解码换成官方采样参数 + 32k 预算:
                  CMExam 严格口径 {baseStrict?.toFixed(1)}% → {v3Strict?.toFixed(1)}%
                  ({decodeGain !== null && decodeGain > 0 ? '+' : ''}{decodeGain?.toFixed(1)}pp),
                  三卷收尾率全部到 {v3Finished?.toFixed(0)}%。
                  三个训练臂仍是 v2 协议下的历史数字,<b>尚未在 v3 下重评</b>——所以下表末行与前四行之间的差,
                  该读成「解码方式的差」,不是「谁的模型更好」。
                </>
              ) : (
                <>协议 v3 的成绩尚未导出,当前表格仍是 v2 协议下的历史数字。</>
              )}
            </p>
            <div className="hero-actions">
              <a className="btn-solid" href="#scores">看成绩对照板</a>
              <a className="btn-ghost" href="#replay">逐题翻查答卷</a>
            </div>
            <div className="stats">
              <div className="stat">
                <div className="v num v3">{v3Strict?.toFixed(1) ?? '—'}%</div>
                <div className="k">正确解码的基座<br />主力卷严格口径</div>
              </div>
              <div className="stat">
                <div className="v num up">
                  {decodeGain !== null && decodeGain > 0 ? '+' : ''}{decodeGain?.toFixed(1) ?? '—'}pp
                </div>
                <div className="k">不训练、只换解码<br />相对存档基座的涨幅</div>
              </div>
              <div className="stat">
                <div className="v num v3">{v3Finished?.toFixed(0) ?? '—'}%</div>
                <div className="k">
                  v3 三卷收尾率<br />
                  v2 下只有 {baseFinished.length ? `${Math.min(...baseFinished).toFixed(0)}–${Math.max(...baseFinished).toFixed(0)}` : '—'}%
                </div>
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
            <h3>评测协议</h3>
            <div className="sub">同一批固定种子抽样卷 · 同一个判分器 · 两套解码并列</div>
            <div className="rows">
              {SET_ORDER.map((s) => {
                const r = cell(BASELINE, s)
                return (
                  <div key={s}>
                    <span>{SET_SHORT[s]}</span>
                    <span className="num">{r ? `n = ${r.n.toLocaleString()}` : '—'}</span>
                  </div>
                )
              })}
              <div><span>解码 v2</span><span className="num">贪心 · 8,192 tokens</span></div>
              <div><span>解码 v3</span><span className="num">官方采样 · 32,768 tokens</span></div>
              <div><span>判分</span><span className="num">规则 + LLM 兜底 · 截断守卫</span></div>
              <div><span>验证器校准</span><span className="num">200 题 · 96.5%</span></div>
              <div><span>去污染</span><span className="num">10-gram 字面查重</span></div>
            </div>
            <p className="note" style={{ marginTop: '0.9rem', marginBottom: 0 }}>
              CMB 官方 test 集不公开答案(防刷榜),故只用其 val 子集作辅助卷;主力卷为 CMExam。
              表里的 v2 基线是 8 月的存档答卷({baseStrict?.toFixed(1)}%);报告主表用的是同机同期复跑的贪心臂(59.6%),
              两者逐题配对无差异(p=0.73),换哪一个都不改变结论。
            </p>
          </div>
        </section>

        <section id="scores">
          <div className="section-head">
            <h2>成绩对照</h2>
            <span className="note">
              {data.meta.runs.length} 个配置 × {SET_ORDER.filter((s) => data.summary.some((r) => r.set === s)).length} 张考卷,同一批题、同一个判分器
            </span>
          </div>
          <p className="sub-note">
            主数字是严格口径;条内竖线为该口径的 Wilson 95% 置信区间;右侧为相对存档基座(原装基座)的涨跌,
            基座落在本格区间内即标注「持平」。条形按 0–100 绝对刻度,不按列归一化。
            涨跌标注只是区间重叠的粗判,配对 McNemar 检验与 Holm/BH 多重比较校正见仓库 <code>reports/usability-v3.md</code>。
          </p>
          <ScoreMatrix summary={data.summary} runs={data.meta.runs} />
        </section>

        <section id="replay">
          <div className="section-head">
            <h2>逐题对照台</h2>
            <span className="note">先看结论,想看推理再展开——「换解码就会了」一栏是本次裁决最直观的证据</span>
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
                  {q.gold && <span className="gold-badge">标准答案 {q.gold}</span>}
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

              <div className="grid" style={{ '--cols': data.meta.runs.length } as CSSProperties}>
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
            上岗前经 200 题校准(一致率 96.5%)。严格口径的逐题标签在 <code>reports/runs/&lt;run&gt;/&lt;set&gt;.usability.jsonl</code>,
            本页每个数字都可以用 <code>medforge.eval.usability --from-tags</code> 复算。
          </div>
          <div>⚠️ 研究与工程实践,输出不构成任何医疗建议。</div>
        </footer>
      </div>
    </>
  )
}
