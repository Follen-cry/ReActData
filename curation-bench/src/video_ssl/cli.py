"""CLI over the upstream-style suite/session lifecycle and video SSL backend."""
import argparse
import json
import os
from pathlib import Path
from .tools.suite import SuiteController
from .tools.evaluation import finalize


def selected_root(args):
    root=args.suite_dir or os.environ.get('VIDEO_SSL_RUN')
    if not root:
        root=Path('.video_ssl_suite').read_text(encoding='utf-8').strip()
    return Path(root)


def execute_suite(suite):
    from .runner.agent_runner import launch
    suite.prepare_resume()
    result=launch(suite)
    if suite.plan['finalize']=='auto':
        suite=SuiteController.from_state_file(suite.root/'suite_state.json')
        result=finalize(suite)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(prog='video-ssl',description='Agent-driven video SSL data construction method')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('init','check','run'):
        tool=sub.add_parser(name)
        tool.add_argument('--prompt',default='run_prompt.md')
        tool.add_argument('--profile')
        if name!='check':
            tool.add_argument('--root','--suite-dir',dest='root')
            tool.add_argument('--new-suite',action='store_true')
    for name in ('plan','next','task','time-budget','preinit','audit','submit','finetune','eval','fail','report','finalize','resume','extend'):
        tool=sub.add_parser(name)
        tool.add_argument('--suite-dir','--run-dir',dest='suite_dir')
        if name in ('audit','submit'): tool.add_argument('--path',required=True)
        if name=='fail': tool.add_argument('--reason',required=True)
        if name=='resume': tool.add_argument('--inspect',action='store_true')
        if name=='extend': tool.add_argument('--iterations',required=True,type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=='check':
            from .core.layout import load_plan
            result=dict(status='configured',plan=load_plan(args.prompt,profile_path=args.profile),note='Static preflight, no GPU/model loading.')
        elif args.command in ('init','run'):
            if os.environ.get('VIDEO_SSL_AGENT')=='1': raise ValueError('The host owns suite creation and Agent launch')
            suite=SuiteController.from_prompt(args.prompt,args.profile,output_root=args.root,new_suite=args.new_suite)
            result=execute_suite(suite) if args.command=='run' else suite.get_run_plan()
        else:
            suite=SuiteController.from_state_file(selected_root(args)/'suite_state.json')
            if args.command in ('finalize','resume','extend') and os.environ.get('VIDEO_SSL_AGENT')=='1':
                raise ValueError('Final test and lifecycle changes belong to the host')
            if args.command=='plan': result=suite.get_run_plan()
            elif args.command=='next': result=suite.begin_next_target()
            elif args.command=='task': result=suite.get_active_session().summarize_task()
            elif args.command=='time-budget': result=suite.get_active_session().get_time_budget()
            elif args.command=='finalize': result=finalize(suite)
            elif args.command=='resume':
                result=dict(status='resumable',plan=suite.prepare_resume(mutate=False)) if args.inspect else execute_suite(suite)
            elif args.command=='extend': result=dict(status='extended',plan=suite.extend(args.iterations))
            elif args.command in ('audit','submit'): result=suite.get_active_session().dispatch(args.command,path=args.path)
            elif args.command in ('finetune','eval'): result=suite.get_active_session().dispatch(args.command)
            elif args.command=='fail': result=suite.fail(args.reason)
            else: result=getattr(suite,args.command)()
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1 if result.get('status') in ('failed','error') else 0
    except Exception as exc:
        print(json.dumps(dict(status='error',error=str(exc)),ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
