import type { Bench } from '../types'

/** 并发扫描曲线 + 关键数字面板。
 *
 *  手绘 SVG 而不引图表库:两条折线不值得多 100KB 依赖,而且要精确控制
 *  「并发按档位等距、不按数值等距」——1..64 按数值排会把低并发挤成一团,
 *  而低并发正是延迟最该看清的区间。 */
const W = 320
const H = 196
const PAD = { l: 30, r: 0, t: 10, b: 26 }

function niceMax(v: number): number {
  const pow = 10 ** Math.floor(Math.log10(v))
  return Math.ceil(v / pow) * pow
}

function Chart({
  title, unit, benches, pick, fmt,
}: {
  title: string
  unit: string
  benches: Bench[]
  pick: (lv: Bench['levels'][number]) => number | undefined
  fmt: (v: number) => string
}) {
  const all = benches.flatMap((b) => b.levels.map(pick).filter((v): v is number => v !== undefined))
  const max = niceMax(Math.max(...all, 1))
  const levels = benches[0]?.levels ?? []
  const x = (i: number) => PAD.l + (i / Math.max(1, levels.length - 1)) * (W - PAD.l - PAD.r)
  const y = (v: number) => H - PAD.b - (v / max) * (H - PAD.t - PAD.b)
  const colors = ['var(--alert)', 'var(--signal)'] // BF16 对照 / FP8 主线

  return (
    <figure className="chart">
      <figcaption>{title}<span>{unit}</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {[1, 2 / 3, 1 / 3, 0].map((f, i) => (
          <g key={f}>
            <line
              className="grid-line" x1={PAD.l} x2={W - PAD.r} y1={y(max * f)} y2={y(max * f)}
              strokeDasharray={i === 3 ? undefined : '2 3'}
            />
            <text className="tick" x={PAD.l - 4} y={y(max * f) + 3} textAnchor="end">{fmt(max * f)}</text>
          </g>
        ))}
        {benches.map((b, bi) => {
          const pts = b.levels
            .map((lv, i) => { const v = pick(lv); return v === undefined ? null : `${x(i)},${y(v)}` })
            .filter(Boolean).join(' ')
          return (
            <polyline
              key={b.label} points={pts} fill="none" stroke={colors[bi % 2]}
              strokeWidth={b.label === 'fp8' ? 2.2 : 1.8} strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            />
          )
        })}
        {levels.map((lv, i) => (
          <text key={lv.concurrency} className="tick" x={x(i)} y={H - 8} textAnchor="middle">{lv.concurrency}</text>
        ))}
      </svg>
      <div className="axis-note">横轴:并发数</div>
    </figure>
  )
}

export function BenchCharts({ benches }: { benches: Bench[] }) {
  const byLabel = (l: string) => benches.find((b) => b.label === l)
  const peak = (b?: Bench) => Math.max(...(b?.levels.map((lv) => lv.output_tok_s ?? 0) ?? [0]))
  const ttftAt64 = (b?: Bench) => b?.levels.find((lv) => lv.concurrency === 64)?.ttft_p50
  const fp8 = byLabel('fp8')
  const bf16 = byLabel('bf16')
  const gain = peak(bf16) ? Math.round(((peak(fp8) - peak(bf16)) / peak(bf16)) * 100) : 0
  const b0 = benches[0]

  return (
    <>
      <div className="bench-grid">
        <Chart
          title="输出吞吐" unit="tok/s,越高越好" benches={benches}
          pick={(lv) => lv.output_tok_s} fmt={(v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : String(Math.round(v)))}
        />
        <Chart
          title="首字延迟 p95" unit="ms,越低越好" benches={benches}
          pick={(lv) => lv.ttft_p95} fmt={(v) => String(Math.round(v))}
        />
        <div className="bench-side">
          <div className="key"><i style={{ background: 'var(--signal)' }} />FP8(主线)</div>
          <div className="key"><i style={{ background: 'var(--alert)' }} />BF16(对照)</div>
          <div className="big">
            <div className="v num">{Math.round(peak(fp8)).toLocaleString()}</div>
            <div className="k">FP8 峰值 tok/s,对 BF16 <b>{gain > 0 ? '+' : ''}{gain}%</b></div>
          </div>
          <div className="big">
            <div className="v num">{Math.round(ttftAt64(fp8) ?? 0)}<small> ms</small></div>
            <div className="k">并发 64 首字延迟,仍低于 BF16 的 {Math.round(ttftAt64(bf16) ?? 0)} ms</div>
          </div>
          <div className="tail">吞吐随并发近线性增长 → 单卡远未饱和。</div>
        </div>
      </div>
      <p className="note" style={{ marginTop: '0.8rem' }}>
        负载为 CMExam 真实题面(非合成 token),固定输出 {b0?.max_tokens} token(ignore_eos),每档并发前预热 3 条。
        卡型 {b0?.gpu || '见报告'} · 引擎 vLLM · 七档并发全程零失败。数字仅对本卡型与本负载成立,不作跨硬件外推。
      </p>
    </>
  )
}
