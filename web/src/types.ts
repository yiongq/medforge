export type RunKey = string

export interface RunMeta {
  key: RunKey
  label: string
  desc: string
  /** 该 run 用的解码协议:v2 = 贪心 8192,v3 = 官方采样 32768 */
  protocol?: 'v2' | 'v3'
}

export interface Answer {
  /** 思考段(可能被截断);思考/结论的切分在导出侧完成 */
  thinking: string
  thinkingChars: number
  thinkingTruncated: boolean
  /** 结论段:永远完整保留(超长时中段省略,头尾都在) */
  conclusion: string
  redacted?: boolean
  /** 原始总字数,如实上报——长度差异本身是实验现象 */
  chars: number
  /** true=判对 false=判错 null=验证器弃权(计错,但单独可见) */
  correct: boolean | null
  method: string | null
}

export interface Question {
  id: string
  set: string
  bucket: string
  bucketLabel: string
  question: string
  options: Record<string, string> | null
  gold: string
  redacted?: boolean
  meta: Record<string, string>
  answers: Record<RunKey, Answer>
}

export interface ScoreRow {
  run: RunKey
  label: string
  set: string
  n: number
  protocol: 'v2' | 'v3'
  /** 宽口径(as-scored):含 LLM 兜底与从复读段刮出的分 */
  acc: number
  ci: [number, number]
  abstain: number
  /** 严格口径 = 写完 ∧ 有结论 ∧ 答对,前台主数字;没跑过 usability 的 run 缺此字段 */
  strict?: number
  strictCi?: [number, number]
  /** 收尾率:输出里写出了 </think> 的比例 */
  finished?: number
}

export interface Replay {
  meta: {
    protocol: string
    protocols?: Record<string, string>
    runs: RunMeta[]
    sets: Record<string, string>
  }
  summary: ScoreRow[]
  questions: Question[]
}

export interface BenchLevel {
  concurrency: number
  requests: number
  failed: number
  ttft_p50?: number
  ttft_p95?: number
  tpot_p50?: number
  tpot_p95?: number
  output_tok_s?: number
  req_per_s?: number
  wall_s?: number
}

export interface Bench {
  label: string
  model: string
  gpu: string
  max_tokens: number
  levels: BenchLevel[]
}
