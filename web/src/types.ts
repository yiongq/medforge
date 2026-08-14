export type RunKey = string

export interface RunMeta {
  key: RunKey
  label: string
  desc: string
}

export interface Answer {
  /** 思考段(可能被截断);思考/结论的切分在导出侧完成 */
  thinking: string
  thinkingChars: number
  thinkingTruncated: boolean
  /** 结论段:永远完整保留(超长时中段省略,头尾都在) */
  conclusion: string
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
  meta: Record<string, string>
  answers: Record<RunKey, Answer>
}

export interface ScoreRow {
  run: RunKey
  label: string
  set: string
  n: number
  acc: number
  ci: [number, number]
  abstain: number
}

export interface Replay {
  meta: {
    protocol: string
    runs: RunMeta[]
    sets: Record<string, string>
  }
  summary: ScoreRow[]
  questions: Question[]
}
