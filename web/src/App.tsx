import { useEffect, useMemo, useState } from 'react'
import type { Bench, Replay } from './types'
import { ScoreBoard } from './components/ScoreBoard'
import { AnswerCard } from './components/AnswerCard'
import { BenchCharts } from './components/BenchCharts'
import { LivePanel } from './components/LivePanel'

const BUCKET_ORDER = ['regression', 'dpo_fix', 'mixed', 'all_wrong', 'all_correct']

export default function App() {
  const [data, setData] = useState<Replay | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [setFilter, setSetFilter] = useState<string>('all')
  const [bucketFilter, setBucketFilter] = useState<string>('all')
  const [idx, setIdx] = useState(0)
  const [benches, setBenches] = useState<Bench[] | null>(null)

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

  // 过滤条件变了就回到第一题,避免下标越界停在空白页
  useEffect(() => setIdx(0), [setFilter, bucketFilter])

  if (err) return <div className="shell"><p className="empty">数据加载失败:{err}</p></div>
  if (!data) return <div className="shell"><p className="empty">加载中…</p></div>

  const q = list[idx]
  const maxChars = q ? Math.max(...Object.values(q.answers).map((a) => a.chars), 1) : 1

  return (
    <div className="shell">
      <header className="masthead">
        <div>
          <div className="tag">MedForge · 回放模式</div>
          <h1>同一道医学考题,四种训练方案怎么答</h1>
        </div>
        <p className="sub">
          Qwen3.5-4B 的后训练消融实验:抄蒸馏教材(两代)与自采样偏好学习,在同一批考卷上并排对照。
          评测协议 {data.meta.protocol}。
        </p>
        <div className="links">
          <a href="https://github.com/yiongq/medforge" target="_blank" rel="noreferrer">代码与报告 ↗</a>
        </div>
      </header>

      <section>
        <div className="section-head">
          <h2>成绩对照</h2>
          <span className="note">须线为 Wilson 95% 置信区间;弃权计错。数字右侧为相对原装基座的涨跌</span>
        </div>
        <ScoreBoard summary={data.summary} runs={data.meta.runs} sets={data.meta.sets} />
      </section>

      <section>
        <div className="section-head">
          <h2>逐题对照台</h2>
          <span className="note">先看结论,想看推理再展开——字数条的长短差异本身就是现象</span>
        </div>

        <div className="controls">
          <div className="chips">
            <span className="chip-label">考卷</span>
            <button className="chip" aria-pressed={setFilter === 'all'} onClick={() => setSetFilter('all')}>全部</button>
            {Object.entries(data.meta.sets).map(([k, label]) => (
              <button key={k} className="chip" aria-pressed={setFilter === k} onClick={() => setSetFilter(k)}>
                {label.split(' · ')[0]}
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
                <span>{data.meta.sets[q.set]?.split(' · ')[0]}</span>
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
        <section>
          <div className="section-head">
            <h2>部署压测</h2>
            <span className="note">同一份权重的 BF16 与 FP8 两档,真实题面负载下的并发扫描</span>
          </div>
          <BenchCharts benches={benches} />
        </section>
      )}

      <section>
        <div className="section-head">
          <h2>现场提问</h2>
          <span className="note">连到一台跑着 vLLM 的机器,用你自己的题考它</span>
        </div>
        <LivePanel />
      </section>

      <footer>
        <div>
          训练数据对全部考卷做过字面去污染(报告见仓库);判分由规则层 + LLM 兜底的验证器完成,
          上岗前经 200 题人工校准(一致率 96.5%)。
        </div>
        <div>⚠️ 研究与工程实践,输出不构成任何医疗建议。</div>
      </footer>
    </div>
  )
}
