#!/usr/bin/env python3
"""Normalized-observation branch; no diagnosis label or correct-action filtering is exposed."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import agent_v2 as raw
from agent_v2 import MODEL_ID,MODEL_REVISION,ACTIONS,NAMES,SYSTEM,candidates,load_jsonl,sha,dump,jsonl,rollout,public_state,teacher_choice,perform

def normalized_state(state):
 checks={}
 c=state['checks'].get('inspect_config')
 if c is not None:
  models=c.get('models');has_models=isinstance(models,list) and len(models)>0
  saved_known=has_models and all('saved' in m for m in models)
  support_known=has_models and all(isinstance(m.get('supported_functions'),list) for m in models)
  assigned_known=has_models and all(isinstance(m.get('functions'),list) for m in models)
  override=c.get('override')
  checks['configuration']={'export_available':c.get('available',False),'model_count':len(models) if has_models else None,
   'all_changes_saved':all(m['saved'] for m in models) if saved_known else None,
   'some_model_supports_seek':any('seek' in m['supported_functions'] for m in models) if support_known else None,
   'some_model_assigned_to_seek':any('seek' in m['functions'] for m in models) if assigned_known else None,
   'override_supplied':bool(override),'override_matches_a_configured_model':override in [m.get('id') for m in models] if override and has_models else None}
 r=state['checks'].get('validate_request')
 if r is not None:checks['request']={'export_available':r.get('available',False),'exactly_one_agent_or_ntl':r['selector_count']==1 if r.get('available') else None,'params_present':r.get('params_present'),'params_is_array':r.get('params_is_array')}
 p=state['checks'].get('parse_response')
 if p is not None:checks['response']={k:p.get(k) for k in ['available','status','envelope_is_json','answer_type','answer_is_application_json','expected_output']}
 return {'area':state['area'],'user':state['user'],'export_names_already_observed':sorted(state['artifacts']),
         'observed_contract_checks':checks,'retrieved_documents':state['documents'],'previous_actions':state['used'],'unavailable_exports':state['missing_exports']}

def prompt_messages(state,gates='standard',style='menu'):
 menu='\n'.join(f'{l}: {NAMES[l]}' for l in candidates(state,gates))
 text='OBSERVED CONTRACT FACTS (computed from supplied exports, not a diagnosis):\n'+json.dumps(normalized_state(state),separators=(',',':'))+'\nPERMITTED NEXT ACTIONS:\n'+menu+'\nNEXT ACTION LETTER:'
 return [{'role':'system','content':SYSTEM},{'role':'user','content':text}]

class LMPolicy(raw.LMPolicy):
 def choose(self,state,gates='standard'):
  allowed=candidates(state,gates);inp=self.tokenizer.apply_chat_template(prompt_messages(state,gates),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.device)
  with self.torch.no_grad():logits=self.model(**inp).logits[0,-1]
  scores={l:float(logits[self.ids[l]]) for l in allowed};chosen=max(scores,key=scores.get)
  return chosen,{'forward_calls':1,'input_tokens':int(inp['input_ids'].shape[-1]),'candidate_count':len(allowed),'candidate_scores':scores,'selected_action_tokens':1,'generated_tokens':0}

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default=MODEL_ID);p.add_argument('--adapter');p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=7);p.add_argument('--gates',choices=['standard','none'],default='standard');args=p.parse_args()
 lm=LMPolicy(args.model,args.adapter,args.device);cases=load_jsonl(args.cases);results=[rollout(c,'lm',lm,args.seed,args.gates) for c in cases];out=Path(args.out);jsonl(out/'traces.jsonl',results)
 s={'cases':len(results),'recipe_groups':len(set(r['recipe_group'] for r in results)),'successes':sum(r['success'] for r in results),
    'known_cases':sum(r['gold']!='insufficient_evidence' for r in results),'known_successes':sum(r['success'] and r['gold']!='insufficient_evidence' for r in results),
    'unknown_cases':sum(r['gold']=='insufficient_evidence' for r in results),'appropriate_selected_handoffs':sum(r['success'] and r['gold']=='insufficient_evidence' for r in results),
    'normal_cases':sum(r['gold']=='no_fault_observed' for r in results),'normal_successes':sum(r['success'] and r['gold']=='no_fault_observed' for r in results),
    'compound_cases':sum(r['gold']=='multiple_issues' for r in results),'compound_successes':sum(r['success'] and r['gold']=='multiple_issues' for r in results),
    'invalid_actions':sum(r['invalid_actions'] for r in results),'forward_calls':sum(r['forward_calls'] for r in results),'input_tokens':sum(r['input_tokens'] for r in results),
    'candidate_scores_computed':sum(r['candidate_scores_computed'] for r in results),'seconds':sum(r['seconds'] for r in results),'case_file_sha256':sha(args.cases),'args':vars(args),
    'representation':'deterministic facts from already observed exports/checks, no gold or diagnosis field; same candidates and evidence gate as raw branch'}
 dump(out/'summary.json',s);print(json.dumps(s))
if __name__=='__main__':main()
