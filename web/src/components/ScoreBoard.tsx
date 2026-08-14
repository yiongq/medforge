import type { ScoreRow, RunMeta } from '../types'

/** 每套考卷一块板:四个方案横向条形图 + Wilson 95% 置信区间须线 + 相对基座的涨跌。
 *  刻意不画折线:四个方案不是时间序列,是四个独立配方。 */
export function ScoreBoard({
  summary, runs, sets,
}: { summary: ScoreRow[]; runs: RunMeta[]; sets: Record<string, string> }) {
  const setKeys = [...new Set(summary.map((s) => s.set))]
  return (
    <div className="boards">
      {setKeys.map((setKey) => {
        const rows = runs
          .map((r) => summary.find((s) => s.set === setKey && s.run === r.key))
          .filter((s): s is ScoreRow => Boolean(s))
        if (!rows.length) return null
        const base = rows.find((r) => r.run === 'base-v2')
        const max = Math.max(...rows.map((r) => r.ci[1]), 30)
        return (
          <div className="board" key={setKey}>
            <h3>{sets[setKey]?.split(' · ')[0] ?? setKey}</h3>
            <div className="board-note">
              {sets[setKey]?.split(' · ')[1] ?? ''} · n={rows[0].n}
            </div>
            {rows.map((r) => {
              const delta = base && r.run !== base.run ? r.acc - base.acc : null
              const pct = (v: number) => `${(v / max) * 100}%`
              return (
                <div className={`bar-row${r.run === 'base-v2' ? ' is-base' : ''}`} key={r.run}>
                  <span className="name">{r.label}</span>
                  <div className="track">
                    <div
                      className={`fill${delta !== null && delta < -1 ? ' is-down' : ''}`}
                      style={{ width: pct(r.acc) }}
                    />
                    <div
                      className="ci"
                      style={{ left: pct(r.ci[0]), width: pct(r.ci[1] - r.ci[0]) }}
                      title={`95% 置信区间 ${r.ci[0]}–${r.ci[1]}%`}
                    />
                  </div>
                  <span className="val num">
                    {r.acc.toFixed(1)}
                    {delta !== null && (
                      <span className={`delta ${delta < 0 ? 'down' : 'up'}`}>
                        {delta > 0 ? '+' : ''}
                        {delta.toFixed(1)}
                      </span>
                    )}
                  </span>
                </div>
              )
            })}
          </div>
        )
      })}
    </div>
  )
}
