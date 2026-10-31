"""Supported backend contract, shared by preflight and the GPU worker."""
import math


def validate(c):
    required={'task','rows_per_trial','max_trials','model_init_seed','train_seed','build_seed','eval_seed','training','evaluation','selection'}
    if set(c) != required:
        raise ValueError('task.contract requires exactly '+', '.join(sorted(required)))
    if c['task'] != 'shape_color_then_scale':
        raise ValueError('Unsupported real task')
    for k in ('rows_per_trial','max_trials'):
        if type(c[k]) is not int or c[k]<1:
            raise ValueError(k+' must be a positive integer')
    if c['max_trials']>7:
        raise ValueError('max_trials exceeds backend limit 7')
    for k in ('model_init_seed','train_seed','build_seed','eval_seed'):
        if type(c[k]) is not int:
            raise ValueError(k+' must be an integer')
    t=c['training']; e=c['evaluation']
    training_keys={'max_steps','global_batch','epochs','shuffle','warmup_steps','scheduler','lora_lr','gen_decoder_lr',
                   'lm_weight','gen_weight','gen_warmup_steps','image_size','checkpoint','restart_from_base_each_trial','timeout_seconds'}
    eval_keys={'samples','test_samples','diffusion_steps','all_cfg_scale','part_cfg_scale','scorer','aggregation',
               'invalid_image_score','dev_timeout_seconds','test_timeout_seconds'}
    if set(t)!=training_keys or set(e)!=eval_keys:
        raise ValueError('Unknown/missing training or evaluation contract fields')
    for name in ('max_steps','global_batch','timeout_seconds'):
        if type(t[name]) is not int or t[name]<1:
            raise ValueError('training.'+name+' must be a positive integer')
    for name in ('samples','test_samples','diffusion_steps','dev_timeout_seconds','test_timeout_seconds'):
        if type(e[name]) is not int or e[name]<1:
            raise ValueError('evaluation.'+name+' must be a positive integer')
    if t['epochs']!=1 or t['shuffle'] is not False or t['restart_from_base_each_trial'] is not True:
        raise ValueError('Backend requires one ordered pass and restart from base')
    if c['rows_per_trial']!=t['max_steps']*t['global_batch']:
        raise ValueError('rows_per_trial must equal max_steps * global_batch')
    if type(t['warmup_steps']) is not int or not 0<=t['warmup_steps']<t['max_steps']:
        raise ValueError('warmup_steps outside training range')
    if t['scheduler']!='linear_warmup_then_cosine' or t['checkpoint']!='last_step' or t['image_size']!=512:
        raise ValueError('Unsupported scheduler/checkpoint/image size')
    for v in [t['lora_lr'],t['gen_decoder_lr'],t['gen_weight'],t['lm_weight'],e['all_cfg_scale'],e['part_cfg_scale']]:
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError('Recipe numerical values must be finite and nonnegative')
    if type(t['gen_warmup_steps']) is not int or not 0<=t['gen_warmup_steps']<=t['max_steps']:
        raise ValueError('invalid gen_warmup_steps')
    if e['scorer']!='analogy_grid_v1_scale' or e['aggregation']!='arithmetic_mean_including_invalid' or e['invalid_image_score']!=0:
        raise ValueError('Unsupported scorer or aggregation')
    if c['selection']!={'metric':'dev.mean','tie_break':'baseline_then_earliest','retrain':False}:
        raise ValueError('Unsupported checkpoint selection rule')
    return c
