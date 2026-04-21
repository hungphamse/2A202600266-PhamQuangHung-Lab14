import json
from collections import Counter, defaultdict

with open('reports/benchmark_results.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)
with open('reports/summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

total = len(entries)
pass_count = sum(1 for e in entries if e.get('status') == 'pass')
fail_count = total - pass_count
pass_rate = pass_count / total if total else 0

metrics = summary.get('metrics', {})
avg_score = metrics.get('avg_score')
faithfulness = metrics.get('faithfulness')
relevancy = metrics.get('relevancy')
hit_rate = metrics.get('hit_rate')
mrr = metrics.get('mrr')
agreement = metrics.get('agreement_rate')

cats = Counter()
cat_examples = defaultdict(list)


def classify(reason, ragas_reason):
    r = (reason or '') + ' ' + (ragas_reason or '')
    rr = r.lower()
    if 'halluc' in rr:
        return 'Hallucination'
    if 'không đủ dữ liệu' in rr or 'insufficient data' in rr or 'not enough data' in rr or 'lack of data' in rr or 'does not contain' in rr or 'does not include' in rr or 'not contain' in rr:
        return 'Incomplete'
    if 'tone' in rr and ('mismatch' in rr or 'tone' in rr and 'not' in rr):
        return 'Tone Mismatch'
    if 'không chính xác' in rr or 'không đúng' in rr or 'incorrect' in rr or 'not accurate' in rr or 'does not reflect' in rr or 'does not match' in rr:
        return 'Incorrect'
    return 'Other'

fails = [e for e in entries if e.get('status') != 'pass']
for e in fails:
    judge = e.get('judge', {}) or {}
    reasoning = judge.get('reasoning', '')
    ragas_reason = (e.get('ragas') or {}).get('reasoning', '')
    cat = classify(reasoning, ragas_reason)
    cats[cat] += 1
    if len(cat_examples[cat]) < 3:
        cat_examples[cat].append({
            'test_case': e.get('test_case'),
            'agent_response': e.get('agent_response'),
            'final_score': judge.get('final_score')
        })

fails_sorted = sorted(fails, key=lambda e: (e.get('judge') or {}).get('final_score', 0))
worst = []
for e in fails_sorted[:3]:
    j = e.get('judge') or {}
    worst.append({
        'test_case': e.get('test_case'),
        'final_score': j.get('final_score'),
        'reasoning': j.get('reasoning')
    })

out = {
    'total': total,
    'pass': pass_count,
    'fail': fail_count,
    'pass_rate': round(pass_rate, 4),
    'avg_score': avg_score,
    'faithfulness': faithfulness,
    'relevancy': relevancy,
    'hit_rate': hit_rate,
    'mrr': mrr,
    'agreement': agreement,
    'clusters': dict(cats),
    'cluster_examples': dict(cat_examples),
    'worst_cases': worst
}

print(json.dumps(out, ensure_ascii=False, indent=2))
