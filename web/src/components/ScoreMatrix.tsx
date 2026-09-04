import type { RunMeta, ScoreRow } from '../types'

/** 各方案 × 三考卷的成绩矩阵。
 *
 *  为什么是矩阵而不是三块独立的板:读者真正要做的比较是「同一张卷子上,各方案谁高谁低」
 *  以及「同一个方案,跨卷是否一致」——矩阵两个方向都能扫,独立板只能扫一个方向。
 *
 *  主数字用严格口径(写完 ∧ 有结论 ∧ 答对):W2 的「降分」大半是没交卷被当成答错,
 *  只报宽口径会把解码工件说成模型差异。宽口径与收尾率放小字里,两个数都在,读者可以自己对。
 *
 *  条形按 0–100 绝对刻度,不按列内最大值归一化:困难卷本来就该看起来矮一截,
 *  归一化会把「25% 和 70% 都画满」的假象喂给读者。 */
const SET_ORDER = ['cmexam', 'cmb-val', 'medxpertqa']
const SET_TITLE: Record<string, { name: string; tag: string }> = {
  cmexam: { name: 'CMExam', tag: '主力卷 · 中文执业医师真题' },
  'cmb-val': { name: 'CMB-val', tag: '辅助卷 · 中文医学综合' },
  medxpertqa: { name: 'MedXpertQA', tag: '困难卷 · 英文 10 选项' },
}
const BASELINE = 'base-v2'

/** 主数字优先严格口径;没跑过 usability 的 run 退回宽口径,并在小字里说明。 */
function scoreOf(row: ScoreRow) {
  const strict = row.strict ?? row.acc
  const ci = row.strictCi ?? row.ci
  return { value: strict, ci, isStrict: row.strict !== undefined }
}

function deltaClass(ciLo: number, ciHi: number, baseValue: number) {
  // 基线落在本格置信区间内 → 视为持平,不涂色报喜也不报忧
  if (baseValue >= ciLo && baseValue <= ciHi) return 'flat'
  return baseValue > ciHi ? 'down' : 'up'
}

export function ScoreMatrix({ summary, runs }: { summary: ScoreRow[]; runs: RunMeta[] }) {
  const sets = SET_ORDER.filter((s) => summary.some((r) => r.set === s))
  const base = (set: string) => summary.find((r) => r.set === set && r.run === BASELINE)
  return (
    <>
      <div className="matrix-wrap">
        <div className="matrix">
          <div className="matrix-row head">
            <div>方案</div>
            {sets.map((s) => (
              <div key={s}>
                {SET_TITLE[s]?.name ?? s}
                <span className="tag"> · {SET_TITLE[s]?.tag} · n={summary.find((r) => r.set === s)?.n.toLocaleString()}</span>
              </div>
            ))}
          </div>
          {runs.map((run) => {
            const isV3 = run.protocol === 'v3'
            const markColor = isV3
              ? 'var(--v3)'
              : run.key === BASELINE
                ? 'var(--ink)'
                : run.key === 'dpo-v2'
                  ? 'var(--signal)'
                  : 'var(--alert)'
            return (
              <div className={`matrix-row${isV3 ? ' v3' : ''}`} key={run.key}>
                <div className="who">
                  <span className="mark" style={{ background: markColor }} />
                  <span>
                    <span className="label">{run.label}</span>
                    {run.protocol && <span className={`proto-tag ${run.protocol}`}>协议 {run.protocol}</span>}
                    <span className="desc" style={{ display: 'block' }}>{run.desc}</span>
                  </span>
                </div>
                {sets.map((s) => {
                  const row = summary.find((r) => r.set === s && r.run === run.key)
                  const b = base(s)
                  if (!row) return <div className="cell" key={s}>—</div>
                  const { value, ci, isStrict } = scoreOf(row)
                  const bv = b ? scoreOf(b).value : null
                  const d = bv === null ? 0 : value - bv
                  const cls = run.key === BASELINE || bv === null ? '' : deltaClass(ci[0], ci[1], bv)
                  return (
                    <div className="cell" key={s}>
                      <div className="top">
                        <span className="acc num">{value.toFixed(1)}%</span>
                        {run.key !== BASELINE && (
                          <span className={`delta num ${cls}`}>
                            {d > 0 ? '+' : ''}{d.toFixed(1)}pp{cls === 'flat' ? ' · 持平' : ''}
                          </span>
                        )}
                      </div>
                      <div className="track">
                        <div
                          className={`fill${isV3 ? ' v3' : cls === 'down' ? ' down' : ''}`}
                          style={{ width: `${value}%` }}
                        />
                        <div
                          className="ci"
                          style={{ left: `${ci[0]}%`, width: `${ci[1] - ci[0]}%` }}
                          title={`95% 置信区间 ${ci[0]}–${ci[1]}%`}
                        />
                      </div>
                      <div className="ci-text num">
                        95% CI {ci[0]}–{ci[1]}
                        {isStrict
                          ? ` · 宽口径 ${row.acc.toFixed(1)}% · 收尾 ${(row.finished ?? 0).toFixed(0)}%`
                          : ' · 仅宽口径'}
                      </div>
                    </div>
                  )
                })}
              </div>
            )
          })}
        </div>
      </div>
      <p className="sub-note matrix-note">
        前四行为协议 v2(贪心、8192 预算),末行为协议 v3(官方采样、32768)。
        严格口径 = 写完 ∧ 有结论 ∧ 答对;宽口径含撞上限后从复读段刮出的分。
      </p>
    </>
  )
}
