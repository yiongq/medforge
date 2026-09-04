# 去污染报告(字面层 · 三通道)

训练题池:54492 条(来源:cmexam-train)
剔除样本:53 条 → 干净题池 54439 条

## 方法:三条通道并列,各自计数、各自可复算

1. `ngram` — 字符 10-gram 倒排索引,评测题 shingle 覆盖率 ≥0.8 判污染、≥0.3 记存疑;模板噪声按文档频率 >0.5% 剔出索引。归一化后 <10 字符的题干直接跳过 —— 这类题在本通道里结构性不可见。
2. `stem_exact` — 归一化题干精确相等。**不受**上面的短题干跳过限制,专补短题。
3. `stem_options_exact` — 归一化「题干 + 按字母序拼接的选项文本」精确相等,即真重题。

归一化三条通道共用一套(去空白/标点/大小写,见 `decontaminate.normalize_text`)。

## 剔除判据按训练源选择

| 训练源 | ngram 污染档 | stem_exact | stem_options_exact |
|---|---|---|---|
| cmexam-train | 仅报告 | 仅报告 | **剔除** |

依据(cmexam-train × 主力卷 2000 题的实测):n-gram 通道判污染 181 题(9.05%)、存疑 194 题(9.70%),
而「题干+选项完全一致且 gold 相同」的真重题只有约 12 道(0.6%);仅题干一致的 104 道里有 92 道选项不同,
是同一考点模板下的不同题。照搬 n-gram 阈值当剔除判据会误剔约 88% 的好数据,所以 cmexam-train 只认
`stem_options_exact` 剔除、`stem_exact` 只报存疑,n-gram 通道保留召回与报告用途。
开放题三源(med-o1-verifiable / med-o1-sft-zh / med-r1-zh)的判据一个字节不改,仍是 n-gram 单通道。

## 三通道数字

计数口径 = 命中的**评测题数**(eval_id 去重);`stem_options_exact 剔除` 列另计被剔的**训练样本数**。

| 评测卷 | 题数 | ngram 污染 | ngram 存疑 | stem_exact 命中 | ├ 选项相同 | └ 选项不同 | stem_options_exact 剔除(训练样本) | 短题干不可扫描 |
|---|---|---|---|---|---|---|---|---|
| cmexam | 6810 | 588(8.63%) | 694(10.19%) | 313 | 48(0.70%) | 265 | 50 | 835 |
| cmb-val | 280 | 8(2.86%) | 10(3.57%) | 6 | 3(1.07%) | 3 | 3 | 21 |
| medxpertqa | 2450 | 0(0.00%) | 0(0.00%) | 0 | 0(0.00%) | 0 | 0 | 0 |
| cmexam-2000(主力卷) | 2000 | 181(9.05%) | 194(9.70%) | 104 | 12(0.60%) | 92 | 12 | 236 |

cmexam-2000(主力卷) = cmexam test 用 `random.Random(42).sample(..., 2000)` 抽的固定卷,
与 `eval/run.py --samples cmexam=2000` 同构;它是 cmexam 全卷的子集,不额外产生剔除,单列一行只为让
上面「判据依据」引用的实测数字能被直接复算。

CMExam test 有 72.10% 题干不足 30 字符(4910/6810),字面近似查重对它们基本无效;精确通道覆盖了其中的完全重合。
其中归一化后不足 10 字符的 835 道在 n-gram 通道里被直接跳过(结构性不可见),而精确通道在这批题里捞出 82 道题干完全相同的训练样本、其中 3 道连选项也完全一致 —— 后者已按判据剔除。

## stem_options_exact 命中清单(真重题,全部;即剔除对象)

- `cmexam-test-53` ← `cmexam-train-35840`(vs cmexam)
- `cmexam-test-110` ← `cmexam-train-11471`(vs cmexam)
- `cmexam-test-180` ← `cmexam-train-31872`(vs cmexam)
- `cmexam-test-259` ← `cmexam-train-36004`(vs cmexam)
- `cmexam-test-289` ← `cmexam-train-13929`(vs cmexam)
- `cmexam-test-341` ← `cmexam-train-31018`(vs cmexam)
- `cmexam-test-526` ← `cmexam-train-16578`(vs cmexam)
- `cmexam-test-805` ← `cmexam-train-867`(vs cmexam)
- `cmexam-test-901` ← `cmexam-train-25625`(vs cmexam)
- `cmexam-test-901` ← `cmexam-train-29679`(vs cmexam)
- `cmexam-test-1008` ← `cmexam-train-23252`(vs cmexam)
- `cmexam-test-1082` ← `cmexam-train-3099`(vs cmexam)
- `cmexam-test-1293` ← `cmexam-train-2135`(vs cmexam)
- `cmexam-test-1310` ← `cmexam-train-197`(vs cmexam)
- `cmexam-test-1545` ← `cmexam-train-35438`(vs cmexam)
- `cmexam-test-1642` ← `cmexam-train-27659`(vs cmexam)
- `cmexam-test-1664` ← `cmexam-train-32651`(vs cmexam)
- `cmexam-test-1835` ← `cmexam-train-28232`(vs cmexam)
- `cmexam-test-2017` ← `cmexam-train-28434`(vs cmexam)
- `cmexam-test-2101` ← `cmexam-train-38004`(vs cmexam)
- `cmexam-test-2381` ← `cmexam-train-1850`(vs cmexam)
- `cmexam-test-2414` ← `cmexam-train-39492`(vs cmexam)
- `cmexam-test-2418` ← `cmexam-train-28834`(vs cmexam)
- `cmexam-test-2444` ← `cmexam-train-35142`(vs cmexam)
- `cmexam-test-2548` ← `cmexam-train-21652`(vs cmexam)
- `cmexam-test-2557` ← `cmexam-train-28430`(vs cmexam)
- `cmexam-test-2625` ← `cmexam-train-28060`(vs cmexam)
- `cmexam-test-2665` ← `cmexam-train-14008`(vs cmexam)
- `cmexam-test-2785` ← `cmexam-train-32257`(vs cmexam)
- `cmexam-test-2835` ← `cmexam-train-21588`(vs cmexam)
- `cmexam-test-2955` ← `cmexam-train-16291`(vs cmexam)
- `cmexam-test-2981` ← `cmexam-train-15668`(vs cmexam)
- `cmexam-test-3232` ← `cmexam-train-2526`(vs cmexam)
- `cmexam-test-3326` ← `cmexam-train-40189`(vs cmexam)
- `cmexam-test-3386` ← `cmexam-train-39494`(vs cmexam)
- `cmexam-test-3415` ← `cmexam-train-27428`(vs cmexam)
- `cmexam-test-3629` ← `cmexam-train-14296`(vs cmexam)
- `cmexam-test-3791` ← `cmexam-train-14712`(vs cmexam)
- `cmexam-test-4238` ← `cmexam-train-28309`(vs cmexam)
- `cmexam-test-4311` ← `cmexam-train-38393`(vs cmexam)
- `cmexam-test-4335` ← `cmexam-train-25809`(vs cmexam)
- `cmexam-test-4374` ← `cmexam-train-2059`(vs cmexam)
- `cmexam-test-4627` ← `cmexam-train-2064`(vs cmexam)
- `cmexam-test-4829` ← `cmexam-train-39772`(vs cmexam)
- `cmexam-test-5314` ← `cmexam-train-29125`(vs cmexam)
- `cmexam-test-6223` ← `cmexam-train-23269`(vs cmexam)
- `cmexam-test-6227` ← `cmexam-train-29131`(vs cmexam)
- `cmexam-test-6521` ← `cmexam-train-28835`(vs cmexam)
- `cmexam-test-6794` ← `cmexam-train-36817`(vs cmexam)
- `cmexam-test-6794` ← `cmexam-train-36857`(vs cmexam)
- `cmb-val-241` ← `cmexam-train-45460`(vs cmb-val)
- `cmb-val-273` ← `cmexam-train-41054`(vs cmb-val)
- `cmb-val-274` ← `cmexam-train-43337`(vs cmb-val)

## stem_exact 存疑清单(题干同、选项不同 → 同模板不同题,不剔;各卷 top 20)

### vs cmexam

- `cmexam-test-4` ← `cmexam-train-49471`
- `cmexam-test-30` ← `cmexam-train-52628`
- `cmexam-test-72` ← `cmexam-train-39299`
- `cmexam-test-76` ← `cmexam-train-10775`
- `cmexam-test-111` ← `cmexam-train-30191`
- `cmexam-test-112` ← `cmexam-train-25288`
- `cmexam-test-132` ← `cmexam-train-27920`
- `cmexam-test-134` ← `cmexam-train-48382`
- `cmexam-test-189` ← `cmexam-train-16627`
- `cmexam-test-261` ← `cmexam-train-43620`
- `cmexam-test-263` ← `cmexam-train-819`
- `cmexam-test-276` ← `cmexam-train-33166`
- `cmexam-test-276` ← `cmexam-train-33286`
- `cmexam-test-276` ← `cmexam-train-33603`
- `cmexam-test-276` ← `cmexam-train-33610`
- `cmexam-test-276` ← `cmexam-train-33697`
- `cmexam-test-276` ← `cmexam-train-33790`
- `cmexam-test-276` ← `cmexam-train-34024`
- `cmexam-test-276` ← `cmexam-train-34256`
- `cmexam-test-309` ← `cmexam-train-35392`

### vs cmb-val

- `cmb-val-16` ← `cmexam-train-45102`
- `cmb-val-214` ← `cmexam-train-46734`
- `cmb-val-252` ← `cmexam-train-45460`

## ngram 存疑清单(覆盖率降序,各卷 top 20,供人工抽看;不作为剔除判据)

### vs cmexam

- `cmexam-test-3363` ← `cmexam-train-37237`(覆盖率 0.798)
- `cmexam-test-3522` ← `cmexam-train-37029`(覆盖率 0.798)
- `cmexam-test-4121` ← `cmexam-train-31213`(覆盖率 0.798)
- `cmexam-test-2421` ← `cmexam-train-51945`(覆盖率 0.797)
- `cmexam-test-2971` ← `cmexam-train-29821`(覆盖率 0.797)
- `cmexam-test-5341` ← `cmexam-train-17716`(覆盖率 0.797)
- `cmexam-test-5637` ← `cmexam-train-30920`(覆盖率 0.797)
- `cmexam-test-1046` ← `cmexam-train-35190`(覆盖率 0.796)
- `cmexam-test-1201` ← `cmexam-train-29988`(覆盖率 0.796)
- `cmexam-test-1201` ← `cmexam-train-29999`(覆盖率 0.796)
- `cmexam-test-2959` ← `cmexam-train-30366`(覆盖率 0.796)
- `cmexam-test-2959` ← `cmexam-train-30397`(覆盖率 0.796)
- `cmexam-test-3434` ← `cmexam-train-37314`(覆盖率 0.796)
- `cmexam-test-4277` ← `cmexam-train-41917`(覆盖率 0.796)
- `cmexam-test-4277` ← `cmexam-train-41847`(覆盖率 0.796)
- `cmexam-test-4277` ← `cmexam-train-41832`(覆盖率 0.796)
- `cmexam-test-150` ← `cmexam-train-29428`(覆盖率 0.795)
- `cmexam-test-3281` ← `cmexam-train-29477`(覆盖率 0.795)
- `cmexam-test-3281` ← `cmexam-train-29246`(覆盖率 0.795)
- `cmexam-test-3912` ← `cmexam-train-52442`(覆盖率 0.795)

### vs cmb-val

- `cmb-val-278` ← `cmexam-train-47786`(覆盖率 0.75)
- `cmb-val-248` ← `cmexam-train-48494`(覆盖率 0.6)
- `cmb-val-258` ← `cmexam-train-43921`(覆盖率 0.556)
- `cmb-val-274` ← `cmexam-train-43180`(覆盖率 0.533)
- `cmb-val-274` ← `cmexam-train-43314`(覆盖率 0.533)
- `cmb-val-274` ← `cmexam-train-43385`(覆盖率 0.533)
- `cmb-val-178` ← `cmexam-train-27649`(覆盖率 0.5)
- `cmb-val-182` ← `cmexam-train-27649`(覆盖率 0.5)
- `cmb-val-235` ← `cmexam-train-43240`(覆盖率 0.5)
- `cmb-val-235` ← `cmexam-train-43296`(覆盖率 0.5)
- `cmb-val-274` ← `cmexam-train-43259`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43244`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43406`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43307`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43370`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43195`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43352`(覆盖率 0.467)
- `cmb-val-274` ← `cmexam-train-43262`(覆盖率 0.467)
- `cmb-val-106` ← `cmexam-train-29048`(覆盖率 0.4)
- `cmb-val-107` ← `cmexam-train-29048`(覆盖率 0.4)

### vs cmexam-2000(主力卷)

- `cmexam-test-3522` ← `cmexam-train-37029`(覆盖率 0.798)
- `cmexam-test-2971` ← `cmexam-train-29821`(覆盖率 0.797)
- `cmexam-test-3434` ← `cmexam-train-37314`(覆盖率 0.796)
- `cmexam-test-1046` ← `cmexam-train-35190`(覆盖率 0.796)
- `cmexam-test-3281` ← `cmexam-train-29477`(覆盖率 0.795)
- `cmexam-test-3281` ← `cmexam-train-29246`(覆盖率 0.795)
- `cmexam-test-2525` ← `cmexam-train-30887`(覆盖率 0.794)
- `cmexam-test-6141` ← `cmexam-train-31903`(覆盖率 0.793)
- `cmexam-test-1440` ← `cmexam-train-52205`(覆盖率 0.792)
- `cmexam-test-1440` ← `cmexam-train-52114`(覆盖率 0.792)
- `cmexam-test-3944` ← `cmexam-train-11234`(覆盖率 0.791)
- `cmexam-test-3944` ← `cmexam-train-11238`(覆盖率 0.791)
- `cmexam-test-3944` ← `cmexam-train-11242`(覆盖率 0.791)
- `cmexam-test-3944` ← `cmexam-train-11236`(覆盖率 0.791)
- `cmexam-test-3522` ← `cmexam-train-37152`(覆盖率 0.79)
- `cmexam-test-4352` ← `cmexam-train-39837`(覆盖率 0.788)
- `cmexam-test-5448` ← `cmexam-train-35695`(覆盖率 0.786)
- `cmexam-test-5588` ← `cmexam-train-41994`(覆盖率 0.783)
- `cmexam-test-690` ← `cmexam-train-27466`(覆盖率 0.783)
- `cmexam-test-690` ← `cmexam-train-27565`(覆盖率 0.783)

归一层丢弃统计:{'cmexam-train': 5, 'cmexam-test': 1}