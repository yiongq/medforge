import type { RunMeta, ScoreRow } from '../types'

/** 四方案 × 三考卷的成绩矩阵。
 *
 *  为什么是矩阵而不是三块独立的板:读者真正要做的比较是「同一张卷子上,四个方案谁高谁低」
 *  以及「同一个方案,跨卷是否一致」——矩阵两个方向都能扫,独立板只能扫一个方向。
 *
 *  条形按 0–100 绝对刻度,不按列内最大值归一化:困难卷本来就该看起来矮一截,
 *  归一化会把「25% 和 70% 都画满」的假象喂给读者。 */
const SET_ORDER = ['cmexam', 'cmb-val', 'medxpertqa']
const SET_TITLE: Record<string, { name: string; tag: string }> = {
  cmexam: { name: 'CMExam', tag: '主力卷 · 中文执业医师真题' },
  'cmb-val': { name: 'CMB-val', tag: '辅助卷 · 中文医学综合' },
  medxpertqa: { name: 'MedXpertQA', tag: '困难卷 · 英文 10 选项' },
}

function deltaClass(d: number, ciLo: number, ciHi: number, baseAcc: number) {
  // 落在基座置信区间内 → 视为持平,不涂色报喜也不报忧
  if (baseAcc >= ciLo && baseAcc <= ciHi) return 'flat'
  return d < 0 ? 'down' : 'up'
}

export function ScoreMatrix({ summary, runs }: { summary: ScoreRow[]; runs: RunMeta[] }) {
  const sets = SET_ORDER.filter((s) => summary.some((r) => r.set === s))
  const base = (set: string) => summary.find((r) => r.set === set && r.run === 'base-v2')
  return (
    <div className="matrix-wrap">
      <div className="matrix">
        <div className="matrix-row head">
          <div>训练方案</div>
          {sets.map((s) => (
            <div key={s}>
              {SET_TITLE[s]?.name ?? s}
              <span className="tag"> · {SET_TITLE[s]?.tag} · n={summary.find((r) => r.set === s)?.n.toLocaleString()}</span>
            </div>
          ))}
        </div>
        {runs.map((run) => (
          <div className="matrix-row" key={run.key}>
            <div className="who">
              <span className="mark" style={{ background: run.key === 'base-v2' ? 'var(--ink)' : run.key === 'dpo-v2' ? 'var(--signal)' : 'var(--alert)' }} />
              <span>
                <span className="label">{run.label}</span>
                <span className="desc" style={{ display: 'block' }}>{run.desc}</span>
              </span>
            </div>
            {sets.map((s) => {
              const row = summary.find((r) => r.set === s && r.run === run.key)
              const b = base(s)
              if (!row) return <div className="cell" key={s}>—</div>
              const d = b ? row.acc - b.acc : 0
              const cls = run.key === 'base-v2' ? '' : deltaClass(d, row.ci[0], row.ci[1], b?.acc ?? 0)
              return (
                <div className="cell" key={s}>
                  <div className="top">
                    <span className="acc num">{row.acc.toFixed(1)}%</span>
                    {run.key !== 'base-v2' && (
                      <span className={`delta num ${cls}`}>
                        {d > 0 ? '+' : ''}{d.toFixed(1)}pp{cls === 'flat' ? ' · 持平' : ''}
                      </span>
                    )}
                  </div>
                  <div className="track">
                    <div className={`fill${cls === 'down' ? ' down' : ''}`} style={{ width: `${row.acc}%` }} />
                    <div className="ci" style={{ left: `${row.ci[0]}%`, width: `${row.ci[1] - row.ci[0]}%` }} title={`95% 置信区间 ${row.ci[0]}–${row.ci[1]}%`} />
                  </div>
                  <div className="ci-text num">
                    95% CI {row.ci[0]}–{row.ci[1]} · 弃权 {row.abstain}%
                  </div>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
