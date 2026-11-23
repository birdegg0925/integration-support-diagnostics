#!/usr/bin/env python3
"""Render diagnostic/handoff drafts from PUBLIC trace fields only; never read gold/fixture."""
import argparse,copy,json
from pathlib import Path
import agent_v2 as lab
URLS={'D_FUNCTIONS':'https://documentation.neuralseek.com/ui/configure/','D_SEEK':'https://api.neuralseek.com/neuralseek.json',
'D_MAISTRO':'https://api.neuralseek.com/neuralseek.json','D_REST':'https://documentation.neuralseek.com/ui/maistro/features/ntl_functions/get_data/','D_PARSE':'https://api.neuralseek.com/neuralseek.json'}
RECOMMENDATIONS={
'function_unassigned':'Have an authorized owner assign Seek to a capable model and save the configuration, then retest.',
'invalid_seek_override':'Compare the request override to the exact configured LLM IDs; use an existing intended ID or remove the override before retesting.',
'unsaved_config':'Have the configuration owner save the intended changes. A successful Test does not save settings.',
'capability_not_supported':'Select a provider/model that supports Seek rather than forcing an unsupported capability.',
'params_not_array':'Provide params as an array of name/value string objects; serialize nested application JSON inside value if intended.',
'answer_is_text':'Parse the HTTP envelope separately from answer. Treat answer as text unless validated application JSON is actually returned.',
'non_json_http_body':'Inspect HTTP status, content type and the upstream response before enabling JSON-to-variable parsing.',
'agent_ntl_conflict':'Specify exactly one of agent or ntl for the intended mAIstro execution.',
'no_fault_observed':'The supplied supported checks found no known fault. This is not a live-service health certificate; reproduce the reported symptom next.',
'multiple_issues':'Several supported contract issues were observed. Coordinate an engineer review; this category does not enumerate a complete causal set.',
'insufficient_evidence':'Obtain the missing sanitized export or response and review the observations before assigning a root cause.'}

def redact(x):
 if isinstance(x,dict):return {k:('[REDACTED]' if any(s in k.lower() for s in ['apikey','api_key','authorization','password','secret']) else redact(v)) for k,v in x.items()}
 if isinstance(x,list):return [redact(v) for v in x]
 return x

def packet(result):
 trace=result.get('trace',[]);state=copy.deepcopy(trace[-1].get('state',{})) if trace else {}
 # A turn-budget exit can occur just after a tool observation. Fold that PUBLIC observation
 # into the draft; never recover it from the fixture or gold label.
 if trace:
  last=trace[-1];obs=last.get('observation');act=last.get('action');arg=last.get('argument')
  if obs is not None and act=='ask':
   state.setdefault('artifacts',{}).update(copy.deepcopy(obs.get('simulated_user_export',{})))
   state['missing_exports']=list(dict.fromkeys(state.get('missing_exports',[])+obs.get('missing',[])))
  elif obs is not None and act=='retrieve':state.setdefault('documents',{})[arg]=copy.deepcopy(obs['retrieved'])
  elif obs is not None and act=='check':state.setdefault('checks',{})[arg]=copy.deepcopy(obs)
 terminal=result['terminal'];a=state.get('artifacts',{});checks=state.get('checks',{})
 required=['inspect_config'] if state.get('area')=='Seek' else ['validate_request','parse_response']
 missing_evidence=[f'check_not_completed:{name}' for name in required if name not in checks or not checks[name].get('available')]
 missing_evidence+=['missing_export:'+x for x in state.get('missing_exports',[])]
 config_check=checks.get('inspect_config',{})
 if config_check.get('available'):
  models=config_check.get('models')
  if not isinstance(models,list) or not models:missing_evidence.append('configuration_models_not_available')
  else:
   for index,model in enumerate(models):
    for field in ['functions','supported_functions','saved']:
     if field not in model:missing_evidence.append(f'model_{index}_missing:{field}')
 if 'request' not in a:missing_evidence.append('request_not_observed')
 citations=[]
 for d in state.get('documents',{}).values():
  for doc_id in d['ids']:citations.append({'id':doc_id,'url':URLS[doc_id]})
 return redact({'draft_only':True,'case_id':result.get('case_id'),'packet_type':'diagnostic' if terminal.get('stop')=='accepted' else 'handoff',
 'initial_user':state.get('user'),'area':state.get('area'),'observed_exports':a,'observed_checks':checks,'document_evidence':citations,
 'decision':terminal,'stop_reason':terminal.get('stop'),'handoff_evidence_available':result.get('handoff_evidence_available'),
 'unconfirmed':list(dict.fromkeys(missing_evidence)),'next_engineer_action':RECOMMENDATIONS.get(terminal.get('diagnosis'),RECOMMENDATIONS['insufficient_evidence']),
 'reproduction_request_draft':copy.deepcopy(a.get('request')),'reproduction_missing_explanation':None if 'request' in a else 'No request body was observed; request a sanitized export first.',
 'source_kind':'Draft from simulated case data and tool observations'})

def main():
 p=argparse.ArgumentParser();p.add_argument('--traces',required=True);p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 for r in lab.load_jsonl(args.traces):lab.dump(out/f"{r['case_id']}-draft.json",packet(r))
 print('packets written',out)
if __name__=='__main__':main()
