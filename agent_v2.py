#!/usr/bin/env python3
"""V2: public-state finite-action LM policy on independently generated simulated recipes."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,random,time
from pathlib import Path
MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
MODEL_REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
ACTIONS={
 'A':('ask','configuration_export',''), 'B':('ask','request_response_export',''),
 'C':('retrieve','seek',''), 'D':('retrieve','maistro',''),
 'E':('check','inspect_config',''), 'F':('check','validate_request',''), 'G':('check','parse_response',''),
 'H':('answer','','function_unassigned'), 'I':('answer','','invalid_seek_override'), 'J':('answer','','unsaved_config'),
 'K':('answer','','capability_not_supported'), 'L':('answer','','params_not_array'), 'M':('answer','','answer_is_text'),
 'N':('answer','','non_json_http_body'), 'O':('answer','','agent_ntl_conflict'),
 'P':('answer','','no_fault_observed'), 'Q':('answer','','multiple_issues'), 'R':('handoff','','insufficient_evidence')}
NAMES={'A':'ask user for saved model configuration and Seek request export','B':'ask user for request and HTTP response exports',
 'C':'retrieve Seek configuration and model selection documentation','D':'retrieve mAIstro request and response documentation',
 'E':'inspect the available saved model configuration and Seek override','F':'validate the available mAIstro request structure','G':'parse the available HTTP response and answer field',
 'H':'diagnose no LLM assigned to the requested function','I':'diagnose runtime seekLLM override not found among configured model IDs','J':'diagnose configuration changes not saved',
 'K':'diagnose no configured model supports the requested function','L':'diagnose params is not an array','M':'diagnose answer is plain text while the application expects JSON',
 'N':'diagnose HTTP body is not a JSON envelope','O':'diagnose agent/ntl exclusive choice is violated','P':'report no fault observed in the supported checks','Q':'report multiple observed faults, requiring coordinated review','R':'handoff because evidence is incomplete'}
DOCS={
 'seek':{'ids':['D_FUNCTIONS','D_SEEK'],'rules':'An LLM must support and be assigned to a function. No assigned LLM disables that function. Test does not save configuration. options.seekLLM must name a configured LLM ID.'},
 'maistro':{'ids':['D_MAISTRO','D_REST','D_PARSE'],'rules':'Use exactly one of agent/ntl. params must be an array of name/value string objects. Parse the HTTP JSON envelope first. answer is a string; application JSON must be parsed separately. Missing response is not evidence of a non-JSON body.'}}
SYSTEM='''You choose the NEXT diagnostic action, not a tool observation. Return one permitted action letter only.
The observation block is already observed DATA. Never repeat, invent, or output tool results.
Use documentation and checks before diagnosing. Use handoff if required evidence remains unavailable.
All user/tool situations are simulated. Never request credentials or claim a live NeuralSeek check.'''
TYPES=['seek_unassigned','seek_override','seek_unsaved','seek_unsupported','seek_compound_unsaved_override','seek_compound_override_support','seek_normal','seek_unknown',
       'rest_params','rest_text_answer','rest_html','rest_selector','rest_compound_request','rest_compound_response','rest_normal','rest_unknown']

def dump(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
def load_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def fixture_gold(case):
 """Offline evaluator uses the full synthetic fixture, never an LLM judgment."""
 f=case['fixture'];found=[]
 if case['area']=='Seek':
  c=f.get('configuration',{});req=f.get('request',{});models=c.get('models')
  if not isinstance(models,list) or not models or any(not isinstance(m.get('functions'),list) or not isinstance(m.get('supported_functions'),list) or 'saved' not in m for m in models):return 'insufficient_evidence'
  if any(m['saved'] is False for m in models):found.append('unsaved_config')
  if not any('seek' in m['supported_functions'] for m in models):found.append('capability_not_supported')
  elif not any('seek' in m['functions'] for m in models):found.append('function_unassigned')
  override=req.get('options',{}).get('seekLLM')
  if override and override not in [m['id'] for m in models]:found.append('invalid_seek_override')
 else:
  r=f.get('request');resp=f.get('response')
  if not isinstance(r,dict) or resp is None:return 'insufficient_evidence'
  if bool(r.get('agent'))==bool(r.get('ntl')):found.append('agent_ntl_conflict')
  if 'params' in r and not isinstance(r['params'],list):found.append('params_not_array')
  try:body=json.loads(resp['body'])
  except (ValueError,TypeError):found.append('non_json_http_body');body=None
  if isinstance(body,dict) and isinstance(body.get('answer'),str) and f.get('expected_output')=='json':
   try:app=json.loads(body['answer']);valid=isinstance(app,(dict,list))
   except ValueError:valid=False
   if not valid:found.append('answer_is_text')
 if len(found)>1:return 'multiple_issues'
 return found[0] if found else 'no_fault_observed'

def make_case(typ,group,variant,index):
 rng=random.Random(1919+TYPES.index(typ)*10000+group*101);area='Seek' if typ.startswith('seek') else 'mAIstro'
 names=rng.sample(['atlas','cedar','maple','orbit','raven','sable','tango','violet','retired'],3)
 if group%4==0:names[0]='retired'
 if area=='Seek':
  n=rng.choice([1,2,3]);models=[]
  for j in range(n):models.append({'id':f'{names[j]}_{rng.randrange(100,999)}','functions':['seek'] if j==0 else rng.choice([['translate'],['maistro'],['translate','maistro']]),
            'supported_functions':['seek','translate','maistro'],'saved':True,'languages':rng.choice([['en'],['en','es'],['en','zh']])})
  conf={'models':models,'profile':rng.choice(['development','staging','sandbox'])}
  req={'question':rng.choice(['How do I configure this feature?','Where is the integration guide?','Can this setup answer from our KB?']), 'options':{}}
  if rng.random()<.5:req['options']['seekLLM']=models[0]['id']
  if typ=='seek_unassigned':
   for m in models:m['functions']=[x for x in m['functions'] if x!='seek']
  if typ in ['seek_override','seek_compound_override_support','seek_compound_unsaved_override']:
   configured_ids={m['id'] for m in models};stem=models[rng.randrange(n)]['id'].rsplit('_',1)[0]
   invalid_id=f'{stem}_{rng.randrange(100,999)}'
   while invalid_id in configured_ids:invalid_id=f'{stem}_{rng.randrange(100,999)}'
   req['options']['seekLLM']=invalid_id
  if typ in ['seek_unsaved','seek_compound_unsaved_override']:models[rng.randrange(n)]['saved']=False
  if typ in ['seek_unsupported','seek_compound_override_support']:
   for m in models:m['functions']=['translate'];m['supported_functions']=['translate','maistro']
   req['options'].pop('seekLLM',None) if typ=='seek_unsupported' else None
  if typ=='seek_unknown':
   for m in models:m.pop('supported_functions')
  f={'configuration':conf,'request':req}
  initial=rng.choice(['Seek still behaves incorrectly after an LLM configuration change.','Please diagnose our Seek model setup before I open a support ticket.','The configured model works in a test, but the Seek path needs review.'])
 else:
  req=({'agent':f'{names[0]}_workflow'} if rng.random()<.6 else {'ntl':'{{ text|value:"integration smoke" }}'})
  req['params']=[{'name':rng.choice(['ticket','payload','question']),'value':json.dumps({'request_id':rng.randrange(1000,9999),'category':rng.choice(['onboarding','billing','access'])})}]
  req['options']={'returnVariables':rng.choice([True,False]),'maxTokens':rng.choice([64,128,192])}
  resp={'status':200,'headers':{'content-type':'application/json'},'body':json.dumps({'answer':json.dumps({'status':'ok','count':rng.randrange(1,9)}),'variables':{'trace':names[1]}})}
  expected='json'
  if typ in ['rest_params','rest_compound_request','rest_compound_response']:req['params']={'payload':{'request_id':rng.randrange(1000,9999)}}
  if typ in ['rest_text_answer']:resp['body']=json.dumps({'answer':rng.choice(['The workflow completed.','No open request was found.','The provided record was accepted.']),'variables':{}})
  if typ in ['rest_html','rest_compound_response']:resp={'status':rng.choice([502,503]),'headers':{'content-type':'text/html'},'body':f'<html><body>{names[0]} upstream unavailable</body></html>'}
  if typ in ['rest_selector','rest_compound_request']:req['agent']=f'{names[0]}_workflow';req['ntl']='Hello integration'
  if typ=='rest_normal' and rng.random()<.5:expected='text';resp['body']=json.dumps({'answer':'The workflow completed normally.','variables':{}})
  if typ=='rest_unknown':resp=None
  f={'request':req,'response':resp,'expected_output':expected}
  initial=rng.choice(['Our mAIstro REST integration needs a request and parser review.','Please inspect the mAIstro request/response contract before the integration is released.',f'The application expects {expected} from mAIstro; help isolate what is wrong.'])
 # Context availability is itself varied; no root-cause flag is exposed.
 supplied=(group+variant)%3==0
 public_artifacts=copy.deepcopy(f) if supplied else {}
 split='train' if group<8 else 'val' if group<10 else 'test'
 case={'case_id':f'V2-{index:04d}','area':area,'scenario_type':typ,'recipe_group':f'{typ}-g{group:02d}','split':split,
       'initial_user':initial+(' I have attached a sanitized export.' if supplied else ' I can provide a sanitized export if needed.'),
       'initial_artifacts':public_artifacts,'fixture':f,'provenance':'Programmatically generated integration scenario'}
 expected={'seek_unassigned':'function_unassigned','seek_override':'invalid_seek_override','seek_unsaved':'unsaved_config','seek_unsupported':'capability_not_supported','seek_compound_unsaved_override':'multiple_issues','seek_compound_override_support':'multiple_issues','seek_normal':'no_fault_observed','seek_unknown':'insufficient_evidence','rest_params':'params_not_array','rest_text_answer':'answer_is_text','rest_html':'non_json_http_body','rest_selector':'agent_ntl_conflict','rest_compound_request':'multiple_issues','rest_compound_response':'multiple_issues','rest_normal':'no_fault_observed','rest_unknown':'insufficient_evidence'}
 case['gold']=expected[typ]
 if fixture_gold(case)!=case['gold']:raise ValueError('fixture does not satisfy independently specified scenario label')
 return case

def public_state(case):return {'area':case['area'],'user':case['initial_user'],'artifacts':copy.deepcopy(case['initial_artifacts']),
                               'checks':{},'documents':{},'used':[],'missing_exports':[],'check_fingerprints':{}}

def check_config(a):
 c=a.get('configuration');req=a.get('request')
 if not isinstance(c,dict) or not isinstance(req,dict):return {'available':False,'missing':['configuration','request']}
 return {'available':True,'models':copy.deepcopy(c.get('models')),'override':req.get('options',{}).get('seekLLM')}
def check_request(a):
 r=a.get('request')
 if not isinstance(r,dict):return {'available':False,'missing':['request']}
 return {'available':True,'selector_count':int(bool(r.get('agent')))+int(bool(r.get('ntl'))),'params_present':'params' in r,
         'params_is_array':isinstance(r.get('params',[]),list),'param_value_types':[type(p.get('value')).__name__ for p in r.get('params',[]) if isinstance(p,dict)] if isinstance(r.get('params',[]),list) else None}
def check_response(a):
 r=a.get('response')
 if not isinstance(r,dict):return {'available':False,'missing':['response']}
 try:envelope=json.loads(r.get('body'));valid=True
 except (ValueError,TypeError):envelope=None;valid=False
 answer=envelope.get('answer') if isinstance(envelope,dict) else None
 try:application=json.loads(answer) if isinstance(answer,str) else None;app_json=isinstance(application,(dict,list))
 except ValueError:app_json=False
 return {'available':True,'status':r.get('status'),'content_type':r.get('headers',{}).get('content-type'),
         'envelope_is_json':valid,'answer_type':type(answer).__name__,'answer_is_application_json':app_json,'expected_output':a.get('expected_output','unknown')}

def perform(case,state,letter):
 action,arg,diag=ACTIONS[letter];state['used'].append(letter)
 if action=='ask':
  wanted=['configuration','request'] if letter=='A' else ['request','response','expected_output'];given={k:copy.deepcopy(case['fixture'][k]) for k in wanted if k in case['fixture']}
  state['artifacts'].update(given)
  absent=[k for k in wanted if k not in given or given[k] is None]
  state['missing_exports']=list(dict.fromkeys(state['missing_exports']+absent));return {'simulated_user_export':given,'missing':absent}
 if action=='retrieve':state['documents'][arg]=copy.deepcopy(DOCS[arg]);return {'retrieved':copy.deepcopy(DOCS[arg])}
 if action=='check':
  fn={'E':check_config,'F':check_request,'G':check_response}[letter];result=fn(state['artifacts']);state['checks'][arg]=result;state['check_fingerprints'][letter]=hashlib.sha256(json.dumps(state['artifacts'],sort_keys=True).encode()).hexdigest();return result
 return {'selected_diagnosis':diag}

def handoff_evidence_available(state):
    relevant='A' if state['area']=='Seek' else 'B'
    if relevant not in state['used']:return False
    required=['configuration','request'] if state['area']=='Seek' else ['request','response','expected_output']
    if any(k in state['missing_exports'] for k in required):return True
    if state['area']=='Seek':
        checked=state['checks'].get('inspect_config')
        if not checked:return False
        models=checked.get('models')
        return not checked.get('available') or not isinstance(models,list) or not models or any(not isinstance(m.get('supported_functions'),list) or not isinstance(m.get('functions'),list) or 'saved' not in m for m in models)
    return any(not v.get('available') for k,v in state['checks'].items() if k in ['validate_request','parse_response'])

def supported_diagnosis(state):
 """Evidence gate diagnoses only observed checks. It never chooses an LM action."""
 issues=[]
 if state['area']=='Seek':
  c=state['checks'].get('inspect_config',{});models=c.get('models')
  if not c.get('available') or not isinstance(models,list) or not models or any(not isinstance(m.get('functions'),list) or not isinstance(m.get('supported_functions'),list) or 'saved' not in m for m in models):return None
  if any(m['saved'] is False for m in models):issues.append('unsaved_config')
  if all('seek' not in m['supported_functions'] for m in models):issues.append('capability_not_supported')
  elif all('seek' not in m['functions'] for m in models):issues.append('function_unassigned')
  if c.get('override') and c['override'] not in [m['id'] for m in models]:issues.append('invalid_seek_override')
 else:
  r=state['checks'].get('validate_request',{});p=state['checks'].get('parse_response',{})
  if not r.get('available') or not p.get('available'):return None
  if r['selector_count']!=1:issues.append('agent_ntl_conflict')
  if r['params_present'] and not r['params_is_array']:issues.append('params_not_array')
  if not p['envelope_is_json']:issues.append('non_json_http_body')
  elif p['answer_type']=='str' and p['expected_output']=='json' and not p['answer_is_application_json']:issues.append('answer_is_text')
 return 'multiple_issues' if len(issues)>1 else issues[0] if issues else 'no_fault_observed'

def candidates(state,gates='standard'):
 if gates=='none':return list(ACTIONS)
 out=[]
 for letter,(act,arg,diag) in ACTIONS.items():
  if act in ['ask','retrieve'] and letter in state['used']:continue
  if act=='check' and state['check_fingerprints'].get(letter)==hashlib.sha256(json.dumps(state['artifacts'],sort_keys=True).encode()).hexdigest():continue
  if act=='answer' and (not state['documents'] or not state['checks']):continue
  out.append(letter)
 return out

def teacher_choice(state,gates='standard'):
 # Observation-only demonstration policy; no fixture or gold argument.
 used=state['used'];area=state['area'];a=state['artifacts'];letters=candidates(state,gates)
 ask='A' if area=='Seek' else 'B';retrieve='C' if area=='Seek' else 'D'
 required=['configuration','request'] if area=='Seek' else ['request','response','expected_output']
 if any(k not in a or a[k] is None for k in required) and ask not in used:return ask
 if any(a.get(k) is None for k in required) and ask in used:return 'R'
 if retrieve not in used:return retrieve
 needed=['E'] if area=='Seek' else ['F','G']
 for l in needed:
  if l not in used:return l
 support=supported_diagnosis(state)
 if support is None:return ask if ask not in used else 'R'
 return next(l for l,v in ACTIONS.items() if v[0]=='answer' and v[2]==support)

def prompt_messages(state,gates='standard',style='menu'):
 allowed=candidates(state,gates)
 # Stored artifacts are shown separately from already computed observations, never as assistant output.
 observation={k:state[k] for k in ['area','user','artifacts','checks','documents','used','missing_exports']}
 menu='\n'.join(f'{l}: {NAMES[l]}' for l in allowed)
 text='OBSERVED DATA (do not output this block):\n'+json.dumps(observation,separators=(',',':'))+'\nPERMITTED NEXT ACTIONS:\n'+menu+'\nNEXT ACTION LETTER:'
 return [{'role':'system','content':SYSTEM},{'role':'user','content':text}]

class LMPolicy:
 def __init__(self,model_path,adapter=None,device='cpu',threads=4):
  import torch
  from transformers import AutoModelForCausalLM,AutoTokenizer
  torch.set_num_threads(threads);self.torch=torch;self.device=device
  self.tokenizer=AutoTokenizer.from_pretrained(model_path,revision=MODEL_REVISION if model_path==MODEL_ID else None);self.model=AutoModelForCausalLM.from_pretrained(model_path,revision=MODEL_REVISION if model_path==MODEL_ID else None,torch_dtype=torch.float32).to(device)
  if adapter:
   from peft import PeftModel
   self.model=PeftModel.from_pretrained(self.model,adapter)
  self.model.eval();self.ids={}
  for l in ACTIONS:
   ids=self.tokenizer(l,add_special_tokens=False)['input_ids']
   if len(ids)!=1:raise ValueError('action label must have exactly one token')
   self.ids[l]=ids[0]
 def choose(self,state,gates='standard'):
  allowed=candidates(state,gates);inp=self.tokenizer.apply_chat_template(prompt_messages(state,gates),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.device)
  with self.torch.no_grad():logits=self.model(**inp).logits[0,-1]
  scores={l:float(logits[self.ids[l]]) for l in allowed};chosen=max(scores,key=scores.get)
  return chosen,{'forward_calls':1,'input_tokens':int(inp['input_ids'].shape[-1]),'candidate_count':len(allowed),'candidate_scores':scores,'selected_action_tokens':1,'generated_tokens':0}

class FreegenPolicy(LMPolicy):
 def __init__(self,*args,max_new_tokens=128,**kwargs):super().__init__(*args,**kwargs);self.max_new_tokens=max_new_tokens
 def choose(self,state,gates='standard'):
  inp=self.tokenizer.apply_chat_template(prompt_messages(state,gates),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(self.device)
  with self.torch.no_grad():out=self.model.generate(**inp,max_new_tokens=self.max_new_tokens,do_sample=False,pad_token_id=self.tokenizer.eos_token_id)
  text=self.tokenizer.decode(out[0,inp['input_ids'].shape[-1]:],skip_special_tokens=True).strip()
  return text if text in candidates(state,gates) else None,{'forward_calls':int(out.shape[-1]-inp['input_ids'].shape[-1]),'input_tokens':int(inp['input_ids'].shape[-1]),'candidate_count':len(candidates(state,gates)),
               'selected_action_tokens':1,'generated_tokens':int(out.shape[-1]-inp['input_ids'].shape[-1]),'raw_output':text}

def rollout(case,policy='rule',lm=None,seed=1,gates='standard',max_turns=8):
 state=public_state(case);rng=random.Random(seed);trace=[];terminal=None;start=time.perf_counter();forward=inputs=scores=0;invalid=0
 for t in range(max_turns):
  before=time.perf_counter();visible=copy.deepcopy(state);usage=None
  try:
   if policy=='rule':letter=teacher_choice(state,gates);usage={'forward_calls':0,'input_tokens':0,'candidate_count':len(candidates(state,gates))}
   elif policy=='random':letter=rng.choice(candidates(state,gates));usage={'forward_calls':0,'input_tokens':0,'candidate_count':len(candidates(state,gates))}
   else:letter,usage=lm.choose(state,gates)
   forward+=usage['forward_calls'];inputs+=usage['input_tokens'];scores+=usage['candidate_count']
   if letter not in candidates(state,gates):raise ValueError('invalid generated action; raw='+str(usage.get('raw_output',''))[:160])
  except Exception as exc:trace.append({'turn':t,'error':type(exc).__name__+':'+str(exc),'state':visible,'usage':usage,'compute_unknown':usage is None});terminal={'action':'handoff','diagnosis':'insufficient_evidence','stop':'policy_failure'};invalid+=1;break
  act,arg,diag=ACTIONS[letter];row={'turn':t,'state':visible,'letter':letter,'action':act,'argument':arg,'diagnosis':diag,'usage':usage,'seconds':time.perf_counter()-before}
  if act in ['ask','retrieve','check']:row['observation']=perform(case,state,letter)
  elif act=='handoff':terminal={'action':'handoff','diagnosis':diag,'stop':'selected_handoff'};trace.append(row);break
  else:
   doc='seek' if state['area']=='Seek' else 'maistro';support=supported_diagnosis(state)
   accepted=doc in state['documents'] and diag==support
   terminal={'action':'answer' if accepted else 'handoff','diagnosis':diag if accepted else 'insufficient_evidence','stop':'accepted' if accepted else 'evidence_rejected','raw_selected_diagnosis':diag}
   trace.append(row);break
  trace.append(row)
 if terminal is None:terminal={'action':'handoff','diagnosis':'insufficient_evidence','stop':'turn_budget'}
 success=(terminal['diagnosis']==case['gold'] and ((case['gold']=='insufficient_evidence' and terminal['stop']=='selected_handoff' and handoff_evidence_available(state)) or (case['gold']!='insufficient_evidence' and terminal['stop']=='accepted')))
 return {'case_id':case['case_id'],'scenario_type':case['scenario_type'],'recipe_group':case['recipe_group'],'gold':case['gold'],'policy':policy,'gates':gates,'success':success,
         'terminal':terminal,'handoff_evidence_available':handoff_evidence_available(state),'turns':len(trace),'forward_calls':forward,'input_tokens':inputs,'candidate_scores_computed':scores,'invalid_actions':invalid,'seconds':time.perf_counter()-start,'trace':trace,
         'data_kind':'simulated new configurations of supported mechanisms; not company incidents'}

def evaluate(args):
 cases=load_jsonl(args.cases)
 if args.limit:cases=cases[:args.limit]
 lm=None
 if args.policy in ['lm','freegen']:lm=(FreegenPolicy if args.policy=='freegen' else LMPolicy)(args.model,args.adapter,args.device)
 results=[rollout(c,args.policy,lm,args.seed,args.gates,args.max_turns) for c in cases]
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);jsonl(out/'traces.jsonl',results)
 types=sorted(set(r['scenario_type'] for r in results));macro=sum(sum(r['success'] for r in results if r['scenario_type']==x)/sum(r['scenario_type']==x for r in results) for x in types)/len(types)
 summary={'cases':len(results),'recipe_groups':len(set(r['recipe_group'] for r in results)),'successes':sum(r['success'] for r in results),'macro_scenario_success':macro,
          'known_cases':sum(r['gold']!='insufficient_evidence' for r in results),'known_successes':sum(r['success'] and r['gold']!='insufficient_evidence' for r in results),
          'unknown_cases':sum(r['gold']=='insufficient_evidence' for r in results),'appropriate_selected_handoffs':sum(r['success'] and r['gold']=='insufficient_evidence' for r in results),
          'normal_cases':sum(r['gold']=='no_fault_observed' for r in results),'normal_successes':sum(r['success'] and r['gold']=='no_fault_observed' for r in results),
          'compound_cases':sum(r['gold']=='multiple_issues' for r in results),'compound_successes':sum(r['success'] and r['gold']=='multiple_issues' for r in results),
          'forward_calls':sum(r['forward_calls'] for r in results),'input_tokens':sum(r['input_tokens'] for r in results),'candidate_scores_computed':sum(r['candidate_scores_computed'] for r in results),
          'invalid_actions':sum(r['invalid_actions'] for r in results),'seconds':sum(r['seconds'] for r in results),'args':vars(args),'case_file_sha256':sha(args.cases),'source_sha256':sha(__file__),
          'protocol':'Known supported mechanisms in held-out configuration recipe groups; not the v1 unseen-mechanism test. Two phrasings per recipe are clustered.'}
 dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False))

def prepare(args):
 cases=[];i=1
 for typ in TYPES:
  for group in range(12):
   for variant in range(2):cases.append(make_case(typ,group,variant,i));i+=1
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 for s in ['train','val','test']:jsonl(out/f'{s}.jsonl',[c for c in cases if c['split']==s])
 rows=[]
 for c in cases:
  if c['split']=='test':continue
  state=public_state(c)
  for t in range(8):
   letter=teacher_choice(state);rows.append({'case_id':c['case_id'],'recipe_group':c['recipe_group'],'scenario_type':c['scenario_type'],'split':c['split'],'state':copy.deepcopy(state),'letter':letter,'teacher':'Fixed rules applied to observed case fields'})
   if ACTIONS[letter][0] in ['answer','handoff']:break
   perform(c,state,letter)
 for s in ['train','val']:jsonl(out/f'policy_{s}.jsonl',[r for r in rows if r['split']==s])
 jsonl(out/'val_small.jsonl',[next(c for c in cases if c['split']=='val' and c['scenario_type']==typ) for typ in TYPES])
 seal={'test_sha256':sha(out/'test.jsonl'),'test_cases':64,'test_recipe_groups':32,'train_cases':256,'val_cases':64,'scenario_types':TYPES,'case_generation_source_sha256':sha(__file__),
       'test_policy':'Do not inspect test outcomes before model/checkpoint selection on validation. Legacy v1 cases are development-only.',
       'grouping':'scenario mechanism × independently sampled configuration recipe; two views share the SAME fixture and differ only in report/context availability',
       'observed_action_demonstrations':{l:sum(r['letter']==l and r['split']=='train' for r in rows) for l in ACTIONS},'company_review':False}
 dump(out/'protocol_seal.json',seal);print(json.dumps(seal,ensure_ascii=False))

def verify(args):
 train=load_jsonl(Path(args.data)/'train.jsonl');val=load_jsonl(Path(args.data)/'val.jsonl');test=load_jsonl(Path(args.data)/'test.jsonl');checks={}
 sets=[set(c['recipe_group'] for c in x) for x in [train,val,test]];checks['recipe_groups_disjoint']=not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
 checks['gold_matches_independent_fixture_rules']=all(fixture_gold(c)==c['gold'] for c in train+val+test)
 c=copy.deepcopy(train[0]);state=public_state(c);a=candidates(state);c['gold']='CANARY';c['fixture']['secret']='PRIVATE_CANARY';checks['candidates_do_not_read_gold']=candidates(state)==a
 checks['no_fixture_or_gold_in_prompt']='PRIVATE_CANARY' not in json.dumps(prompt_messages(state)) and 'CANARY' not in json.dumps(prompt_messages(state))
 results=[rollout(c,'rule') for c in val];checks['teacher_solves_all_validation']=all(r['success'] for r in results)
 checks['all_terminal_and_tool_classes_demonstrated']=set(r['letter'] for r in load_jsonl(Path(args.data)/'policy_train.jsonl'))==set(ACTIONS)
 recovery=public_state(next(c for c in train if c['area']=='Seek' and not c['initial_artifacts']));case=next(c for c in train if c['area']=='Seek' and not c['initial_artifacts']);perform(case,recovery,'E');blocked_before='E' not in candidates(recovery);perform(case,recovery,'A');checks['recheck_after_new_export_allowed']=blocked_before and 'E' in candidates(recovery)
 checks['immediate_handoff_not_appropriate']=not handoff_evidence_available(public_state(next(c for c in val if c['gold']=='insufficient_evidence')))
 checks['test_hash_matches_seal']=sha(Path(args.data)/'test.jsonl')==json.loads((Path(args.data)/'protocol_seal.json').read_text())['test_sha256']
 dump(args.out,{'checks':checks,'all_passed':all(checks.values())});print(json.dumps(checks))
 if not all(checks.values()):raise SystemExit(1)

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare');a.add_argument('--out',default='data')
 a=sub.add_parser('verify');a.add_argument('--data',default='data');a.add_argument('--out',default='runs/verify.json')
 a=sub.add_parser('evaluate');a.add_argument('--cases',required=True);a.add_argument('--out',required=True);a.add_argument('--policy',choices=['rule','random','lm','freegen'],default='lm');a.add_argument('--model',default=MODEL_ID);a.add_argument('--adapter');a.add_argument('--device',default='cpu');a.add_argument('--seed',type=int,default=7);a.add_argument('--gates',choices=['standard','none'],default='standard');a.add_argument('--max-turns',type=int,default=8);a.add_argument('--limit',type=int,default=0)
 args=p.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
