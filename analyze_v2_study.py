#!/usr/bin/env python3
"""Recompute a frozen simulated study from episode traces; no model inference."""
import argparse, collections, hashlib, json, statistics
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed_missing(state):
    area = state['area']
    if ('A' if area == 'Seek' else 'B') not in state['used']:
        return False
    required = {'configuration', 'request'} if area == 'Seek' else {'request', 'response', 'expected_output'}
    if required.intersection(state['missing_exports']):
        return True
    if area == 'Seek':
        check = state['checks'].get('inspect_config')
        if check is None:
            return False
        models = check.get('models')
        return (not check.get('available') or not isinstance(models, list) or not models
                or any(not isinstance(m.get('functions'), list)
                       or not isinstance(m.get('supported_functions'), list)
                       or 'saved' not in m for m in models))
    return any(not value.get('available') for key, value in state['checks'].items()
               if key in {'validate_request', 'parse_response'})


def recompute(rows, cases):
    expected_ids = set(cases)
    assert len(rows) == len(expected_ids), 'Missing or repeated episodes'
    assert {r['case_id'] for r in rows} == expected_ids, 'Different test cohort'
    category = collections.defaultdict(lambda: [0, 0])
    stops = collections.Counter()
    forwards = tokens = invalid = 0
    successes = known_successes = appropriate = 0
    for row in rows:
        case = cases[row['case_id']]
        assert row['gold'] == case['gold'], 'Stored label differs from frozen case'
        assert row['recipe_group'] == case['recipe_group']
        terminal = row['terminal']
        final = next(t['state'] for t in reversed(row['trace']) if 'state' in t)
        if case['gold'] == 'insufficient_evidence':
            good = (terminal['stop'] == 'selected_handoff'
                    and terminal['diagnosis'] == case['gold']
                    and observed_missing(final))
            appropriate += int(good)
        else:
            good = terminal['stop'] == 'accepted' and terminal['diagnosis'] == case['gold']
            known_successes += int(good)
        assert bool(row['success']) == bool(good), 'Stored success cannot be re-derived'
        calls = sum(t.get('usage', {}).get('forward_calls', 0)
                    for t in row['trace'] if t.get('usage') is not None)
        n_tokens = sum(t.get('usage', {}).get('input_tokens', 0)
                       for t in row['trace'] if t.get('usage') is not None)
        assert calls == row['forward_calls'], 'Failed or successful forwards were omitted'
        assert n_tokens == row['input_tokens']
        forwards += calls
        tokens += n_tokens
        invalid += row['invalid_actions']
        successes += int(good)
        stops[terminal['stop']] += 1
        key = ('unknown' if case['gold'] == 'insufficient_evidence' else
               'normal' if case['gold'] == 'no_fault_observed' else
               'multiple' if case['gold'] == 'multiple_issues' else 'single_fault')
        category[key][0] += int(good)
        category[key][1] += 1
        for step in row['trace']:
            usage = step.get('usage') or {}
            scores = usage.get('candidate_scores')
            if row['policy'] not in {'random', 'rule'} and scores and 'letter' in step:
                assert step['letter'] == max(scores, key=scores.get), 'Policy action was overwritten'
    return {'cases': len(rows), 'recipe_groups': len({c['recipe_group'] for c in cases.values()}),
            'successes': successes, 'known_successes': known_successes,
            'appropriate_handoffs': appropriate, 'category': dict(category),
            'stop_reasons': dict(stops), 'forward_calls': forwards, 'input_tokens': tokens,
            'invalid_actions': invalid,
            'seconds': sum(r['seconds'] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--run-map', required=True,
                        help='JSON groups: [{label,repeat_axis,runs:[folder,...]}]')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    case_rows = read_jsonl(args.cases)
    cases = {r['case_id']: r for r in case_rows}
    assert len(cases) == len(case_rows)
    map_path = Path(args.run_map)
    groups = json.loads(map_path.read_text())['groups']
    results = []
    for group in groups:
        records = []
        for name in group['runs']:
            folder = Path(name)
            if not folder.is_absolute():
                folder = map_path.parent / folder
            summary = json.loads((folder / 'summary.json').read_text())
            assert summary['case_file_sha256'] == digest(args.cases), 'Cohort hash mismatch'
            rows = read_jsonl(folder / 'traces.jsonl')
            record = recompute(rows, cases)
            assert record['successes'] == summary['successes']
            record['run'] = str(folder)
            records.append(record)
        numbers = [r['successes'] for r in records]
        result = {'label': group['label'], 'repeat_axis': group['repeat_axis'],
                  'runs': records, 'mean_successes': statistics.mean(numbers),
                  'min_successes': min(numbers), 'max_successes': max(numbers),
                  'n': len(cases), 'mean_forwards': statistics.mean(r['forward_calls'] for r in records),
                  'mean_input_tokens': statistics.mean(r['input_tokens'] for r in records)}
        results.append(result)
        print(group['label'], numbers, flush=True)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    payload = {'scope': 'Reference study of simulated integration-support cases',
               'case_sha256': digest(args.cases), 'cases': len(cases),
               'recipe_groups': len({r['recipe_group'] for r in case_rows}),
               'configuration_views_are_not_independent': True,
               'interval_note': 'Min/max across repeated fits or random draws; not confidence intervals.',
               'groups': results}
    (output / 'recomputed_results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    lines = ['# 锁定模拟测试：逐轨迹复算结果', '',
             f"同一测试文件：{len(cases)}条任务视图，{payload['recipe_groups']}个配置组。", '',
             '| 方法 | 各次符合验收条件数 | 分母 | 重复轴 | 平均模型前向数 |',
             '|---|---|---|---|---|']
    for result in results:
        lines.append('| ' + result['label'] + ' | ' + ', '.join(str(r['successes']) for r in result['runs'])
                     + f" | {result['n']} | {result['repeat_axis']} | {result['mean_forwards']:.1f} |")
    lines += ['', '总数包含限定诊断、支持范围内未见异常、多问题类别识别，以及取证后适当交接；不等同自动修复故障。',
              '', '重复训练和随机策略的重复轴不同。这里仅列实测次数及范围，不把64条视图或多次随机选择当作更多独立配置。']
    (output / '逐轨迹复算结果.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(max(9, len(results) * 1.2), 4.2))
    positions = list(range(len(results)))
    means = [r['mean_successes'] for r in results]
    lower = [r['mean_successes'] - r['min_successes'] for r in results]
    upper = [r['max_successes'] - r['mean_successes'] for r in results]
    axis.bar(positions, means, yerr=[lower, upper], capsize=4, color='#537c9d')
    axis.set_xticks(positions, [r['label'] for r in results], rotation=25, ha='right')
    axis.set_ylabel('Tasks meeting simulated acceptance criteria')
    axis.set_ylim(0, max(len(cases), max(means)) * 1.15)
    for position, result in zip(positions, results):
        value = result['mean_successes']
        shown = str(int(value)) if float(value).is_integer() else f"{value:.2f}"
        prefix = "mean " if len(result['runs']) > 1 else ""
        axis.text(position, result['max_successes'] + 1, f"{prefix}{shown}/{result['n']}", ha='center')
    axis.set_title('Reference only: same frozen configurations; ranges are not confidence intervals', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'frozen_test_comparison.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
