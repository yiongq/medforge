import type { Bench } from '../types'

/** 并发扫描曲线。手绘 SVG 而不引图表库:两条折线的需求不值得多 100KB 依赖,
 *  而且这样能精确控制「并发按档位等距、不按数值等距」——否则 1..64 的横轴
 *  会把低并发挤成一团,而低并发正是延迟最该看清的区间。 */
const W = 520
const H = 190
const PAD = { l: 46, r: 12, t: 12, b: 28 }

function Line({
  bench, pick, color, max,
}: { bench: Bench; pick: (lv: Bench['levels'][number]) => number | undefined; color: string; max: number }) {
  const pts = bench.levels
    .map((lv, i) => {
      const v = pick(lv)
      if (v === undefined) return null
      const x = PAD.l + (i / Math.max(1, bench.levels.length - 1)) * (W - PAD.l - PAD.r)
      const y = H - PAD.b - (v / max) * (H - PAD.t - PAD.b)
      return { x, y, v }
    })
    .filter((p): p is { x: number; y: number; v: number } => p !== null)
  if (!pts.length) return null
  return (
    <g>
      <polyline
        points={pts.map((p) => `${p.x},${p.y}`).join(' ')}
        fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round"
      />
      {pts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="3" fill={color} />)}
    </g>
  )
}

function Chart({
  title, unit, benches, pick, colors,
}: {
  title: string
  unit: string
  benches: Bench[]
  pick: (lv: Bench['levels'][number]) => number | undefined
  colors: string[]
}) {
  const all = benches.flatMap((b) => b.levels.map(pick).filter((v): v is number => v !== undefined))
  const max = Math.max(...all, 1) * 1.15
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => ({ f, v: max * f }))
  const levels = benches[0]?.levels ?? []
  return (
    <figure className="chart">
      <figcaption>{title}<span className="unit">{unit}</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {ticks.map((t) => {
          const y = H - PAD.b - t.f * (H - PAD.t - PAD.b)
          return (
            <g key={t.f}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y} y2={y} className="grid" />
              <text x={PAD.l - 6} y={y + 3} className="tick" textAnchor="end">
                {t.v >= 100 ? Math.round(t.v) : t.v.toFixed(1)}
              </text>
            </g>
          )
        })}
        {levels.map((lv, i) => {
          const x = PAD.l + (i / Math.max(1, levels.length - 1)) * (W - PAD.l - PAD.r)
          return <text key={lv.concurrency} x={x} y={H - 9} className="tick" textAnchor="middle">{lv.concurrency}</text>
        })}
        {benches.map((b, i) => <Line key={b.label} bench={b} pick={pick} color={colors[i % colors.length]} max={max} />)}
      </svg>
      <div className="legend">
        {benches.map((b, i) => (
          <span key={b.label}>
            <i style={{ background: colors[i % colors.length] }} />
            {b.label.toUpperCase()}
          </span>
        ))}
        <span className="axis-note">横轴:并发路数</span>
      </div>
    </figure>
  )
}

export function BenchCharts({ benches }: { benches: Bench[] }) {
  const colors = ['var(--signal)', 'var(--alert)']
  const b0 = benches[0]
  return (
    <>
      <div className="charts">
        <Chart title="输出吞吐" unit="tok/s" benches={benches} colors={colors} pick={(lv) => lv.output_tok_s} />
        <Chart title="首 token 延迟 p95" unit="ms" benches={benches} colors={colors} pick={(lv) => lv.ttft_p95} />
        <Chart title="每 token 间隔 p50" unit="ms" benches={benches} colors={colors} pick={(lv) => lv.tpot_p50} />
      </div>
      <p className="note">
        负载为 CMExam 真实题面(非合成 token),固定输出 {b0?.max_tokens} token,每档并发前预热 3 条。
        卡型 {b0?.gpu || '见报告'} · 引擎 vLLM。数字仅对本卡型与本负载成立,不作跨硬件外推。
      </p>
    </>
  )
}
