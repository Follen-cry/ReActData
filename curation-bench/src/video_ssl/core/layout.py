"""Resolve prompt -> task, machine profile, runner, optional Docker, protocol.

This is the only input schema for new runs. The backend's flat input is generated,
never maintained by hand alongside these files.
"""
from copy import deepcopy
from datetime import datetime, timezone
from importlib.resources import files
import math
from pathlib import Path
import runpy
import shutil

import yaml
from .prompt import parse_run_prompt
from .runtime import read, digest, resolve, keys

REPO=Path(__file__).resolve().parents[3]


def mapping(path, allowed, required=()):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    keys(value,allowed,str(path))
    if not set(required)<=set(value):
        raise ValueError(f'Missing fields in {path}: {sorted(set(required)-set(value))}')
    return value


def reference(base, name, directory, suffix):
    candidate=resolve(base,name)
    if candidate.is_file():
        return candidate
    candidate=directory/(name if name.endswith(suffix) else name+suffix)
    if candidate.is_file():
        return candidate.resolve()
    raise ValueError(f'Reference not found: {name}')


def positive(value, name):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError(name+' must be positive and finite')
    return value


def load_sources(path):
    sources={}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        import json
        row=json.loads(line)
        if not isinstance(row,dict): raise ValueError('source manifest rows must be objects')
        row=deepcopy(row)
        sid=row.pop('source_video_id',None)
        if not isinstance(sid,str) or not sid or sid in sources:
            raise ValueError('source_video_id must be a unique nonempty string')
        required={'video','f0','final','frame_count','description','eval_text'}
        if not required<=set(row) or set(row)-required-{'frames'}:
            raise ValueError('Invalid source manifest fields')
        if type(row['frame_count']) is not int or row['frame_count']<2:
            raise ValueError('frame_count must be >=2')
        for key in ('video','f0','final'):
            p=resolve(path.parent,row[key])
            if not p.is_file(): raise ValueError(f'Missing source {key}: {p}')
            row[key]=str(p)
        if not isinstance(row.get('frames',{}),dict): raise ValueError('frames must be a mapping')
        bank={}
        for index,value in row.get('frames',{}).items():
            if not isinstance(index,str) or not index.isdecimal() or str(int(index))!=index or not 0<int(index)<row['frame_count']-1:
                raise ValueError('Invalid intermediate frame index')
            p=resolve(path.parent,value)
            if not p.is_file(): raise ValueError('Missing frame PNG: '+str(p))
            bank[index]=str(p)
        row['frames']=bank
        for key in ('description','eval_text'):
            if not isinstance(row[key],str) or not row[key].strip(): raise ValueError('Missing source text')
        sources[sid]=row
    if len({s['video'] for s in sources.values()})!=len(sources):
        raise ValueError('Source IDs must not alias the same video')
    return sources


def load_evaluation(path,count,split):
    import json
    result=[]; ids=set()
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip(): continue
        row=json.loads(line)
        if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id'] or row['id'] in ids:
            raise ValueError(split+' evaluation IDs must be unique strings')
        ids.add(row['id'])
        if not isinstance(row.get('image'),list) or len(row['image'])!=1 or not isinstance(row.get('target_image'),str):
            raise ValueError(split+' evaluation requires one F0 image and one target_image')
        if not isinstance(row.get('conversations'),list) or not row['conversations']:
            raise ValueError(split+' evaluation requires conversations')
        def image(value):
            p=resolve(path.parent,value)
            if not p.is_file(): raise ValueError('Missing evaluation image: '+str(p))
            return str(p)
        row['image']=[image(row['image'][0])]
        row['target_image']=image(row['target_image'])
        row['cond_image']=image(row.get('cond_image',row['image'][0]))
        if row['cond_image']!=row['image'][0]:
            raise ValueError('Frozen F0 evaluation must use the first image as cond_image')
        result.append(row)
    if len(result)!=count:
        raise ValueError(f'{split} manifest count {len(result)} differs from task count {count}')
    return result


def load_plan(prompt_path, *, task_id=None, profile_path=None):
    prompt_path=Path(prompt_path).resolve()
    prompt=parse_run_prompt(prompt_path)
    if len(set(prompt.tasks)) != len(prompt.tasks):
        raise ValueError('Duplicate task IDs are not allowed')
    if task_id is None and len(prompt.tasks)>1:
        plans={name:load_plan(prompt_path, task_id=name, profile_path=profile_path) for name in prompt.tasks}
        ledger=read(next(iter(plans.values()))['campaign'])
        if len(ledger['training_attempts'])+len(plans)*prompt.iterations>ledger['training_budget']:
            raise ValueError('Insufficient shared training attempts for task queue')
        return dict(kind='multi_task',task_order=prompt.tasks,tasks=plans,iterations=prompt.iterations,
                    finalize='manual',campaign=next(iter(plans.values()))['campaign'])
    task_id=task_id or prompt.tasks[0]
    if task_id not in prompt.tasks:
        raise ValueError('Task is not declared in run_prompt')
    if not task_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in task_id):
        raise ValueError('Task ID must be a name, not a path')
    task_dir=resolve(prompt_path.parent,prompt.tasks_dir) if prompt.tasks_dir else REPO/'tasks'
    task_path=task_dir/(task_id+'.yaml')
    task_fields={'task_id','goal','dataset_id','sources_manifest','model_key','training_config','resources',
                 'dev_id','test_id','strategy_timeout_seconds','estimated_trial_seconds','preinit','finalize',
                 'frame_policy','max_dataset_bytes','max_text_chars','contract'}
    task=mapping(task_path,task_fields,task_fields)
    if task['task_id']!=task_id: raise ValueError('task_id does not match task filename')
    backend=REPO/'vendor'/'video_ssl_backend'
    contract=deepcopy(task['contract'])
    runpy.run_path(str(backend/'recipe.py'))['validate'](contract)
    if type(prompt.iterations) is not int or not 1<=prompt.iterations<=contract['max_trials']:
        raise ValueError('iterations includes baseline and must fit task.contract.max_trials')
    if not (profile_path or prompt.profile) or not prompt.campaign:
        raise ValueError('run_prompt requires profile and campaign references')
    profile_path=reference(prompt_path.parent,str(profile_path or prompt.profile),REPO/'profiles','.yaml')
    profile_fields={'python','dataset_path_map','model_path_map','eval_path_map','resource_path_map','gpu_allocation','env'}
    profile=mapping(profile_path,profile_fields,profile_fields-{'env'})
    maps={}
    for category in ('dataset_path_map','model_path_map','eval_path_map','resource_path_map'):
        raw=profile[category]
        if not isinstance(raw,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in raw.items()):
            raise ValueError(category+' must map resource IDs to paths')
        maps[category]={k:resolve(profile_path.parent,v) for k,v in raw.items()}
    def resource(category,key):
        if key not in maps[category]: raise ValueError(f'Unmapped {category} ID: {key}')
        return maps[category][key]
    runner_path=reference(prompt_path.parent,prompt.runner or 'runner',REPO/'configs','.yaml')
    runner_file=mapping(runner_path,{'mode','agent','timeout_seconds','cli_python','docker'},{'mode','agent'})
    mode=runner_file['mode']
    if mode not in ('native','docker'): raise ValueError('runner.mode must be native or docker')
    agent=deepcopy(runner_file['agent'])
    keys(agent,{'executable','model','reasoning_effort'},'runner.agent')
    if not isinstance(agent.get('executable'),str) or not agent['executable']:
        raise ValueError('runner.agent.executable required')
    if mode=='native' and not shutil.which(agent['executable'],path=profile.get('env',{}).get('PATH')):
        raise ValueError('Agent executable not found')
    agent['timeout_seconds']=positive(task['strategy_timeout_seconds'],'strategy_timeout_seconds')
    runner={k:runner_file[k] for k in ('mode','timeout_seconds','cli_python') if k in runner_file}
    if 'timeout_seconds' in runner: positive(runner['timeout_seconds'],'runner.timeout_seconds')
    inputs={'prompt':prompt_path,'task':task_path,'profile':profile_path,'runner':runner_path}
    if mode=='docker':
        docker_path=reference(runner_path.parent,runner_file.get('docker','docker.yaml'),REPO/'configs','.yaml')
        docker_fields={'image','cli_python','read_only_mounts','credential_files','pass_env','mount_profile_resources','env'}
        docker=mapping(docker_path,docker_fields,{'image'})
        if not isinstance(docker['image'],str) or not docker['image']: raise ValueError('Docker image required')
        if set(runner)&set(docker): raise ValueError('Duplicate Docker/runner setting')
        extra=[]
        for raw in docker.get('read_only_mounts',[]): extra.append(str(resolve(docker_path.parent,raw)))
        if docker.get('mount_profile_resources',True):
            extra += [str(p) for category,paths in maps.items() for p in paths.values()
                      if category!='resource_path_map' or p.exists()]
            # Eval rows reference images adjacent to the manifest, not just that file.
            extra += [str(p.parent) for p in maps['eval_path_map'].values()]
        for path in extra:
            if not Path(path).exists(): raise ValueError('Missing Docker mount: '+path)
            if Path(path)==Path(Path(path).anchor): raise ValueError('Do not mount a filesystem root')
        credentials={str(resolve(docker_path.parent,k)):v for k,v in docker.get('credential_files',{}).items()}
        container_env=docker.get('env',{})
        if not isinstance(container_env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in container_env.items()):
            raise ValueError('Docker env must map names to strings')
        if {'HOME','PYTHONPATH','CUDA_VISIBLE_DEVICES','VIDEO_SSL_RUN','VIDEO_SSL_AGENT','VIDEO_SSL_CURRENT_TASK'}&set(container_env):
            raise ValueError('Docker env cannot override method control variables')
        if set(container_env)&set(profile.get('env',{})):
            raise ValueError('Duplicate environment setting in profile and Docker config')
        runner.update({k:v for k,v in docker.items() if k not in ('mount_profile_resources','read_only_mounts','credential_files','env')})
        runner['container_env']=container_env
        runner.update(read_only_mounts=sorted(set(extra)),credential_files=credentials)
        inputs['docker']=docker_path
    # Native mode deliberately never resolves, reads or hashes the Docker reference.
    protocol_path=reference(prompt_path.parent,prompt.protocol or 'method',REPO/'protocols','.md')
    inputs['protocol']=protocol_path
    if protocol_path.parent==REPO/'protocols' and protocol_path.name in ('instruction.md','skill.md'):
        inputs['protocol_base']=REPO/'protocols'/'method.md'
    campaign_path=resolve(prompt_path.parent,prompt.campaign)
    ledger=read(campaign_path)
    if type(ledger['max_concurrent_gpus']) is not int or not 1<=ledger['max_concurrent_gpus']<=4:
        raise ValueError('Campaign max_concurrent_gpus must be 1..4')
    gpu=profile['gpu_allocation']
    batch=contract['training']['global_batch']
    if (not isinstance(gpu,list) or not gpu or any(type(d) is not int or d<0 for d in gpu)
            or len(set(gpu))!=len(gpu) or len(gpu)>min(4,ledger['max_concurrent_gpus']) or batch%len(gpu)):
        raise ValueError('Invalid GPU allocation for global batch')
    if gpu!=ledger['gpu_allocation']: raise ValueError('Profile GPU allocation differs from campaign')
    if type(ledger['training_budget']) is not int or not 1<=ledger['training_budget']<=7:
        raise ValueError('Invalid shared training budget')
    if any(a['status'] in ('preparing','running') for a in ledger['training_attempts']):
        raise ValueError('Campaign has an active training attempt; inspect first')
    if len(ledger['training_attempts'])+prompt.iterations>ledger['training_budget']:
        raise ValueError('Insufficient shared training attempts')
    deadline=datetime.fromisoformat(ledger['deadline'].replace('Z','+00:00'))
    if deadline.tzinfo is None: raise ValueError('Campaign deadline needs a timezone')
    if type(ledger['report_reserve_seconds']) is not int or ledger['report_reserve_seconds']<1800:
        raise ValueError('Campaign reserve must be >=1800 seconds')
    estimate=positive(task['estimated_trial_seconds'],'estimated_trial_seconds')
    if (deadline-datetime.now(timezone.utc)).total_seconds()<estimate+ledger['report_reserve_seconds']:
        raise ValueError('Insufficient campaign time')
    if type(task['preinit']) is not bool or task['finalize'] not in ('manual','auto'):
        raise ValueError('Invalid task preinit/finalize setting')
    frame=task['frame_policy']
    runpy.run_path(str(backend/'frame_policy.py'))['policy']({'frame_policy':frame})
    for k in ('max_dataset_bytes','max_text_chars'):
        if type(task[k]) is not int or task[k]<1: raise ValueError(k+' must be positive integer')
    manifest_name=Path(task['sources_manifest'])
    if manifest_name.is_absolute() or '..' in manifest_name.parts:
        raise ValueError('task.sources_manifest must be relative to the dataset resource')
    sources_path=resolve(resource('dataset_path_map',task['dataset_id']),str(manifest_name))
    sources=load_sources(sources_path)
    if len(sources)<contract['rows_per_trial']: raise ValueError('Source pool smaller than rows_per_trial')
    inputs['sources']=sources_path
    python=resolve(profile_path.parent,profile['python'])
    if mode=='native' and not python.is_file(): raise ValueError('GPU Python missing')
    ext={'weights':str(resource('model_path_map',task['model_key'])),
         'dev_manifest':str(resource('eval_path_map',task['dev_id'])),
         'test_manifest':str(resource('eval_path_map',task['test_id']))}
    keys(task['resources'],{'model_library','prompt_wrapper','scorer_cvlib'},'task.resources')
    if set(task['resources'])!={'model_library','prompt_wrapper','scorer_cvlib'}: raise ValueError('Missing task.resources')
    ext.update({k:str(resource('resource_path_map',v)) for k,v in task['resources'].items()})
    for k,path in ext.items():
        if k=='model_library' and mode=='docker': continue
        if not Path(path).exists(): raise ValueError('Missing external '+k+': '+path)
        if k in ('weights','model_library') and not Path(path).is_dir():
            raise ValueError(k+' must be a directory')
        if k in ('dev_manifest','test_manifest','prompt_wrapper','scorer_cvlib') and not Path(path).is_file():
            raise ValueError(k+' must be a file')
    if ext['dev_manifest']==ext['test_manifest']: raise ValueError('dev and test must be separate')
    evaluation_inputs={split:load_evaluation(Path(ext[split+'_manifest']),number,split)
                       for split,number in [('dev',contract['evaluation']['samples']),('test',contract['evaluation']['test_samples'])]}
    env=profile.get('env',{})
    if not isinstance(env,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in env.items()):
        raise ValueError('Profile env must map names to strings')
    if {'CUDA_VISIBLE_DEVICES','VIDEO_SSL_RUN','VIDEO_SSL_AGENT','VIDEO_SSL_CURRENT_TASK','PYTHONPATH'}&set(env):
        raise ValueError('Profile env cannot override GPU or method control variables')
    training_path=resolve(task_path.parent,task['training_config'])
    training=yaml.safe_load(training_path.read_text(encoding='utf-8'))
    substitutions={'__MODEL_PATH__':ext['weights'],
                   '__MAX_STEPS__':contract['training']['max_steps'],'__GLOBAL_BATCH__':contract['training']['global_batch'],
                   '__LORA_LR__':contract['training']['lora_lr'],'__GEN_DECODER_LR__':contract['training']['gen_decoder_lr'],
                   '__LM_WEIGHT__':contract['training']['lm_weight'],'__GEN_WEIGHT__':contract['training']['gen_weight'],
                   '__GEN_WARMUP_STEPS__':contract['training']['gen_warmup_steps'],
                   '__WARMUP_RATIO__':contract['training']['warmup_steps']/contract['training']['max_steps'],
                   '__MODEL_INIT_SEED__':contract['model_init_seed'],'__EVAL_SEED__':contract['eval_seed'],
                   '__DIFFUSION_STEPS__':contract['evaluation']['diffusion_steps'],
                   '__ALL_CFG__':contract['evaluation']['all_cfg_scale'],'__PART_CFG__':contract['evaluation']['part_cfg_scale']}
    def substitute(value):
        if isinstance(value,dict): return {k:substitute(v) for k,v in value.items()}
        if isinstance(value,list): return [substitute(v) for v in value]
        if isinstance(value,str) and value in substitutions: return substitutions[value]
        if value=='__TRAIN_META__':
            meta=resource('resource_path_map','training_meta')
            if not meta.is_file(): raise ValueError('Missing training_meta resource')
            inputs['training_meta']=meta
            return str(meta)
        if isinstance(value,str) and value.startswith('__') and value.endswith('__'):
            raise ValueError('Unknown training template placeholder: '+value)
        return value
    training=substitute(training)
    if resolve(training_path.parent,training['model']['snapshot'])!=Path(ext['weights']):
        raise ValueError('Training template snapshot differs from model resource')
    t=contract['training']
    # Recipe values have one owner: the task contract. Reject disagreement rather than hide it.
    expected={'train':{'max_steps':t['max_steps'],'global_batch_size':t['global_batch'],'epochs':1,'seed':contract['model_init_seed']},
              'optim':{'lr':t['lora_lr'],'gen_decoder_lr':t['gen_decoder_lr']},
              'loss':{'lm_weight':t['lm_weight'],'gen_weight':t['gen_weight'],'gen_warmup_steps':t['gen_warmup_steps']},
              'gen_image':{'size':t['image_size']},
              'eval':{'diffusion_steps':contract['evaluation']['diffusion_steps'],'seed':contract['eval_seed'],
                      'all_cfg_scale':contract['evaluation']['all_cfg_scale'],'part_cfg_scale':contract['evaluation']['part_cfg_scale']}}
    for section,values in expected.items():
        for key,value in values.items():
            if training.get(section,{}).get(key)!=value:
                raise ValueError(f'Training template {section}.{key} disagrees with task contract')
    inputs['training']=training_path
    inputs['dev']=Path(ext['dev_manifest']); inputs['test']=Path(ext['test_manifest'])
    inputs['prompt_wrapper']=Path(ext['prompt_wrapper']); inputs['scorer']=Path(ext['scorer_cvlib'])
    frozen=set(inputs.values())|set(backend.glob('*.py'))|set(Path(__file__).parent.parent.rglob('*.py'))
    for name in ('pyproject.toml','uv.lock','.python-version'):
        path=REPO/name
        if not path.is_file(): raise ValueError('Missing uv project input: '+name+'; prepare with uv sync --locked')
        frozen.add(path)
        inputs['uv_'+name.replace('.','_')]=path
    frozen |= {Path(path) for rows in evaluation_inputs.values() for row in rows
               for path in row['image']+[row['target_image'],row['cond_image']]}
    backend_input={'mode':'real','task':contract['task'],'task_contract':contract,'sources':sources,'external':ext,
                   'campaign':str(campaign_path),'estimated_trial_seconds':estimate,'frame_policy':frame,
                   'max_dataset_bytes':task['max_dataset_bytes'],'max_text_chars':task['max_text_chars']}
    return {'task_id':task_id,'task_definition':task,'pipeline_dir':str(backend),'python':str(python),
            'iterations':prompt.iterations,'strategy':prompt.strategy_text,'agent':agent,'runner':runner,
            'preinit':task['preinit'],'finalize':task['finalize'],'campaign':str(campaign_path),'env':env,
            'protocol_file':str(protocol_path),'frame_policy':frame,'backend_input':backend_input,'training_input':training,
            'evaluation_inputs':evaluation_inputs,
            'input_files':{k:str(v) for k,v in inputs.items()},'hashes':{str(p):digest(p) for p in frozen}}
