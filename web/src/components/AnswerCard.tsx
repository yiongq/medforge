import { useState } from 'react'
import type { Answer, RunMeta } from '../types'
import { extractDeclared, fmtChars, verdictOf } from '../lib'

/** 一个方案的作答卡。设计要点:
 *  - 结论置顶、思考默认折叠——多栏并排时,先让人看到「答了什么」再看「怎么想的」
 *  - 字数条是长度差异的可视化证据:抄过笔记的模型思考显著变短
 *  - 协议 v3 的卡片挂一个标记:它和同行的差异首先来自解码方式,不是权重
 *  - 思考/结论在导出侧已切分,这里不做二次解析(截断只发生在思考段) */
export function AnswerCard({
  run, answer, maxChars,
}: { run: RunMeta; answer: Answer | undefined; maxChars: number }) {
  const [open, setOpen] = useState(false)
  if (!answer) {
    return (
      <div className="card skip">
        <div className="card-head">
          <div className="who"><h3>{run.label}</h3>{run.protocol === 'v3' && <span className="proto-tag">协议 v3</span>}</div>
          <div className="desc">{run.desc}</div>
        </div>
        <div className="card-body"><p className="empty">该方案未作答此题</p></div>
      </div>
    )
  }
  const v = verdictOf(answer.correct)
  const declared = extractDeclared(answer.conclusion)
  const hasThinking = answer.thinkingChars > 0
  return (
    <div className={`card ${v.tone}`}>
      <div className="card-head">
        <div className="who">
          <h3>{run.label}</h3>
          {run.protocol === 'v3' && <span className="proto-tag">协议 v3</span>}
          <span className={`badge ${v.tone}`}>{v.label}</span>
        </div>
        <div className="desc">{run.desc}</div>
        <div className="card-metrics">
          {declared && <span className="declared mono">选 {declared}</span>}
          <span className="num" title="本次作答总字数(思考+结论)">{fmtChars(answer.chars)}</span>
          <span className="len-track" aria-hidden>
            <span className="len-fill" style={{ width: `${Math.min(100, (answer.chars / maxChars) * 100)}%` }} />
          </span>
        </div>
      </div>
      <div className="card-body">
        <div className="conclusion">
          {answer.redacted
            ? '（按 MedXpertQA 许可不公开作答原文,只展示判分与字数）'
            : answer.conclusion || '（模型未给出结论段——思考未收尾即到达长度上限）'}
        </div>
        {hasThinking && (
          <>
            <button className="thinking-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
              {open ? '收起思考过程' : `展开思考过程 · ${fmtChars(answer.thinkingChars)}`}
            </button>
            {open && (
              <div className="thinking">
                {answer.thinking}
                {answer.thinkingTruncated && (
                  <p className="truncated">（思考过程过长已截断,完整原文见仓库 reports/runs/）</p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
