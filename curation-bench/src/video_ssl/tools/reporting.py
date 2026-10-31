"""Framework-owned reports, code differences and selection differences."""
import json
from ..core.runtime import read, write, selection_delta


def write_report(suite):
    table = ['# Video SSL Data Construction Experiment', '', 'This run develops a data-construction program using fixed downstream dev feedback.', '',
             '| Round | State | dev.mean | Details |', '|---|---|---:|---|']
    tsv = ['iteration\tdev_mean\tstatus\tcode_sha256\tdescription']
    best = None
    previous_code = {}
    previous_rows = []
    if suite.state.get('preinit_result'):
        table.append(f"| preinit | completed | {suite.state['preinit_result']['dev']['mean']} | [base dev](preinit/dev/) |")
        tsv.append(f"0\t{suite.state['preinit_result']['dev']['mean']}\treference\t\tuntrained base dev reference")
    for target in suite.state['targets']:
        directory = suite.target_dir(target)
        score = target.get('dev', {}).get('mean')
        table.append(f"| {target['iteration']} | {target['state']} | {score} | [{target['key']}]({target['key']}/REPORT.md) |")
        if not directory.exists():
            continue
        code = {str(p.relative_to(directory / 'code')):p.read_text(encoding='utf-8') for p in (directory / 'code').rglob('*.py')}
        base_key=target.get('base_target')
        base_code=previous_code
        base_rows=previous_rows
        if base_key:
            base_directory=suite.root/base_key
            base_code={str(p.relative_to(base_directory/'code')):p.read_text(encoding='utf-8')
                       for p in (base_directory/'code').rglob('*.py')}
            base_manifest=base_directory/'curated'/'final_submission'/'manifest.jsonl'
            base_rows=[json.loads(line) for line in base_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        import difflib
        diff = ''.join(''.join(difflib.unified_diff(base_code.get(name,'').splitlines(True), code.get(name,'').splitlines(True),
                     fromfile=(base_key or 'initial')+'/'+name, tofile='current/'+name)) for name in sorted(set(base_code)|set(code)))
        (directory / 'changes.diff').write_text(diff, encoding='utf-8')
        manifest = directory / 'curated' / 'final_submission' / 'manifest.jsonl'
        current = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()] if manifest.exists() else []
        write(directory / 'selection_diff.json', selection_delta(base_rows, current))
        hypothesis = directory / 'code' / 'hypothesis.md'
        description = hypothesis.read_text(encoding='utf-8') if hypothesis.exists() else 'Seeded random video selection; F0-to-final baseline' if target['iteration']==1 else 'Not submitted'
        (directory / 'REPORT.md').write_text(f"# {target['key']}\n\n{description}\n\nState: {target['state']}; dev.mean: {score}\n\n"
            f"Strategy base: {target.get('base_target')}; decision: {target.get('strategy_result')}\n\nError: {target.get('error')}\n\n[Code](code/) · [Code diff](changes.diff) · [Video/frame/order diff](selection_diff.json) · [Data](curated/final_submission/) · [Dev feedback](dev/)\n", encoding='utf-8')
        if target['state'] in ('completed','failed'):
            keep = target['state'] == 'completed' and (best is None or score > best)
            if keep:
                best = score
            label = 'keep' if keep else 'discard' if target['state']=='completed' else 'crash'
            clean = description.replace('\t',' ').replace('\n',' ')[:250]
            tsv.append(f"{target['iteration']}\t{score if score is not None else ''}\t{label}\t{target.get('code_sha256','')}\t{clean}")
        if code:
            previous_code, previous_rows = code, current
    (suite.root / 'results.tsv').write_text('\n'.join(tsv)+'\n', encoding='utf-8')
    table += ['', f"Search status: {suite.state['status']}; final test: {suite.state['final']}.", '',
              '[Configuration](config/plan.json) · [State](suite_state.json) · [Agent logs](output/) · [Final result](final/result.json)']
    (suite.root / 'README.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    return {'status':suite.state['status'], 'report':str(suite.root/'README.md')}
