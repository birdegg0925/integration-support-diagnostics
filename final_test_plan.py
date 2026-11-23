#!/usr/bin/env python3
"""Lock the final test matrix from validation-selected checkpoints before test inference."""
import json,datetime
from pathlib import Path
import agent_v2 as lab

def main():
 facts=Path('models/facts_seed7/training_summary.json')
 if not facts.exists():raise SystemExit('Facts branch has not completed validation; test matrix is not released yet.')
 seal=json.load(open('data/protocol_seal.json'));assert lab.sha('data/test.jsonl')==seal['test_sha256']
 selected={str(s):json.load(open(f'models/seed{s}/training_summary.json'))['selected_step'] for s in [7,11,19]}
 fstep=json.load(open(facts))['selected_step']
 runs=[{'name':'test_raw_base','entry':'agent_v2.py','policy':'lm','gates':'standard','adapter':None}]
 for seed in [7,11,19]:runs.append({'name':f'test_raw_seed{seed}_step{selected[str(seed)]}','entry':'agent_v2.py','policy':'lm','gates':'standard','adapter':f'models/seed{seed}/step_{selected[str(seed)]}'})
 runs += [{'name':'test_raw_base_nogates','entry':'agent_v2.py','policy':'lm','gates':'none','adapter':None},
          {'name':f'test_raw_seed19_step{selected["19"]}_nogates','entry':'agent_v2.py','policy':'lm','gates':'none','adapter':f'models/seed19/step_{selected["19"]}'},
          {'name':'test_facts_base','entry':'agent_facts_v2.py','gates':'standard','adapter':None},
          {'name':f'test_facts_seed7_step{fstep}','entry':'agent_facts_v2.py','gates':'standard','adapter':f'models/facts_seed7/step_{fstep}'}]
 obj={'locked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_sha256':seal['test_sha256'],'test_cases':64,'recipe_groups':32,
      'known_cases':56,'unknown_cases':8,'normal_cases':8,'compound_cases':16,'raw_selected_steps':selected,'facts_selected_step':fstep,
      'neural_runs':runs,'non_neural_runs':{'rule':'test_rule','random':[f'test_random_seed{s:02d}' for s in range(20)]},
      'selection':'Validation only. Raw3seed main, facts single-seed representation side branch. Head-only/1.5B not selected for final test.',
      'post_test_rule':'No modification of method, representation, labels or checkpoint after test inference.'}
 p=Path('test_selection_manifest.json')
 if p.exists():
  old=json.load(open(p))
  for k in ['test_sha256','raw_selected_steps','facts_selected_step','neural_runs']:
   if old[k]!=obj[k]:raise RuntimeError('Attempt to change locked final matrix')
  print('existing final matrix unchanged');return
 lab.dump(p,obj);print(json.dumps(obj,ensure_ascii=False))
if __name__=='__main__':main()
