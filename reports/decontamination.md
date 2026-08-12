# 去污染报告(字面层)

方法:字符 10-gram 倒排索引,评测题 shingle 覆盖率 ≥0.8 判污染(训练侧剔除)、
≥0.3 记存疑(仅报告);模板噪声按文档频率 >0.5% 剔出索引。
只比题干,不比选项。embedding 语义层接入后在此追加第二节。

训练题池:60815 条(来源:med-o1-verifiable, med-o1-sft-zh)
剔除污染样本:17 条 → 干净题池 60798 条

| 评测集 | 题数 | 污染命中 | 存疑命中 | 短题干不可扫描 |
|---|---|---|---|---|
| cmexam | 6810 | 20 | 213 | 835 |
| cmb-val | 280 | 0 | 7 | 21 |
| medxpertqa | 2450 | 0 | 3 | 0 |

「短题干不可扫描」= 归一化后不足 10 字符的题干(如「甘味的作用特点是」),题意在选项中,
对开放题训练池无字面泄漏面;字面层对其无能为力,如实公布计数。

## 存疑清单(覆盖率降序,各集 top 20,供人工抽看)

### vs cmexam

- `cmexam-test-6664` ← `med-o1-sft-zh-7524`(覆盖率 0.792)
- `cmexam-test-5203` ← `med-o1-sft-zh-7779`(覆盖率 0.778)
- `cmexam-test-2431` ← `med-o1-sft-zh-7315`(覆盖率 0.766)
- `cmexam-test-6015` ← `med-o1-sft-zh-14987`(覆盖率 0.761)
- `cmexam-test-5799` ← `med-o1-sft-zh-10037`(覆盖率 0.76)
- `cmexam-test-2621` ← `med-o1-sft-zh-6197`(覆盖率 0.75)
- `cmexam-test-2122` ← `med-o1-sft-zh-15179`(覆盖率 0.742)
- `cmexam-test-3744` ← `med-o1-sft-zh-7462`(覆盖率 0.73)
- `cmexam-test-6681` ← `med-o1-sft-zh-12767`(覆盖率 0.722)
- `cmexam-test-3274` ← `med-o1-sft-zh-7639`(覆盖率 0.686)
- `cmexam-test-2473` ← `med-o1-sft-zh-4705`(覆盖率 0.684)
- `cmexam-test-3257` ← `med-o1-sft-zh-16988`(覆盖率 0.684)
- `cmexam-test-3257` ← `med-o1-sft-zh-19744`(覆盖率 0.684)
- `cmexam-test-5328` ← `med-o1-sft-zh-13278`(覆盖率 0.671)
- `cmexam-test-6274` ← `med-o1-sft-zh-14461`(覆盖率 0.667)
- `cmexam-test-5487` ← `med-o1-sft-zh-4065`(覆盖率 0.664)
- `cmexam-test-613` ← `med-o1-sft-zh-7639`(覆盖率 0.663)
- `cmexam-test-1642` ← `med-o1-sft-zh-17360`(覆盖率 0.656)
- `cmexam-test-5437` ← `med-o1-sft-zh-19494`(覆盖率 0.65)
- `cmexam-test-5437` ← `med-o1-sft-zh-5053`(覆盖率 0.65)

### vs cmb-val

- `cmb-val-274` ← `med-o1-sft-zh-5744`(覆盖率 0.533)
- `cmb-val-274` ← `med-o1-sft-zh-7959`(覆盖率 0.467)
- `cmb-val-235` ← `med-o1-sft-zh-16988`(覆盖率 0.333)
- `cmb-val-235` ← `med-o1-sft-zh-3070`(覆盖率 0.333)
- `cmb-val-235` ← `med-o1-sft-zh-20096`(覆盖率 0.333)
- `cmb-val-235` ← `med-o1-sft-zh-2763`(覆盖率 0.333)
- `cmb-val-235` ← `med-o1-sft-zh-19744`(覆盖率 0.333)

### vs medxpertqa

- `medxpertqa-1620` ← `med-o1-verifiable-5343`(覆盖率 0.366)
- `medxpertqa-1492` ← `med-o1-verifiable-19282`(覆盖率 0.35)
- `medxpertqa-2191` ← `med-o1-verifiable-6217`(覆盖率 0.307)

归一层丢弃统计:{'cmexam-test': 1}