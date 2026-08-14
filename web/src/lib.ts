/** 从结论里找答案声明,用于在卡片头部高亮显示。抽不到就返回 null(与验证器口径一致:宁缺勿猜)。 */
export function extractDeclared(conclusion: string): string | null {
  const m = conclusion.match(/(?:最终答案|正确答案|答案)\s*(?:是|为|[::])?\s*([A-J](?:\s*[、,，和\s]\s*[A-J])*)(?![A-Za-z0-9])/)
  if (m) return m[1].replace(/[\s、,，和]/g, '')
  const en = conclusion.match(/(?:the answer is|answer\s*:)\s*\(?([A-J])\)?/i)
  return en ? en[1].toUpperCase() : null
}

export function verdictOf(correct: boolean | null): { label: string; tone: 'ok' | 'bad' | 'skip' } {
  if (correct === true) return { label: '答对', tone: 'ok' }
  if (correct === false) return { label: '答错', tone: 'bad' }
  return { label: '未给结论', tone: 'skip' }
}

export function fmtChars(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k 字` : `${n} 字`
}
