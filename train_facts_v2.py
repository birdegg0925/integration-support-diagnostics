#!/usr/bin/env python3
"""Balanced next-action-token LoRA supervision; validation-only checkpoint selection."""
import argparse,collections,copy,hashlib,json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model,TaskType
import agent_facts_v2 as lab

def teacher_forced(model,tokenizer,rows,device='cpu',gates='standard'):
 model.eval();by_action=collections.defaultdict(list);terminal=[];inputs=0
 ids={l:tokenizer(l,add_special_tokens=False)['input_ids'][0] for l in lab.ACTIONS}
 with torch.no_grad():
  for r in rows:
   inp=tokenizer.apply_chat_template(lab.prompt_messages(r['state'],gates),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(device)
   logits=model(**inp).logits[0,-1];allowed=lab.candidates(r['state'],gates);pred=max(allowed,key=lambda l:float(logits[ids[l]]));correct=pred==r['letter'];inputs+=int(inp['input_ids'].shape[-1])
   by_action[r['letter']].append(correct)
   if lab.ACTIONS[r['letter']][0] in ['answer','handoff']:terminal.append(correct)
 return {'examples':len(rows),'macro_action_accuracy':sum(sum(v)/len(v) for v in by_action.values())/len(by_action),
         'overall_accuracy':sum(sum(v) for v in by_action.values())/len(rows),'terminal_accuracy':sum(terminal)/len(terminal),
         'per_action':{l:{'correct':sum(v),'n':len(v)} for l,v in by_action.items()},'input_tokens':inputs,'forward_calls':len(rows)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data');p.add_argument('--out',required=True);p.add_argument('--model',default=lab.MODEL_ID);p.add_argument('--seed',type=int,default=7);p.add_argument('--steps',type=int,default=240);p.add_argument('--checkpoints',default='120,240');p.add_argument('--r',type=int,default=8);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
 args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(args.seed);random.seed(args.seed);data=Path(args.data);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 tr=lab.load_jsonl(data/'policy_train.jsonl');va=lab.load_jsonl(data/'policy_val.jsonl');val_cases=lab.load_jsonl(data/'val.jsonl')
 if set(r['recipe_group'] for r in tr)&set(r['recipe_group'] for r in va):raise ValueError('recipe leakage')
 tokenizer=AutoTokenizer.from_pretrained(args.model,revision=lab.MODEL_REVISION if args.model==lab.MODEL_ID else None);model=AutoModelForCausalLM.from_pretrained(args.model,revision=lab.MODEL_REVISION if args.model==lab.MODEL_ID else None,torch_dtype=torch.float32)
 model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,target_modules=['q_proj','v_proj'],r=args.r,lora_alpha=2*args.r,lora_dropout=.05,bias='none')).to(args.device)
 ids={l:tokenizer(l,add_special_tokens=False)['input_ids'][0] for l in lab.ACTIONS}
 encoded=[];groups=collections.defaultdict(list)
 for i,r in enumerate(tr):
  inp=tokenizer.apply_chat_template(lab.prompt_messages(r['state']),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
  encoded.append(inp);groups[r['letter']].append(i)
 if set(groups)!=set(lab.ACTIONS):raise ValueError('every action class needs demonstrations')
 rng=random.Random(args.seed);order=[]
 while len(order)<args.steps:
  letters=list(lab.ACTIONS);rng.shuffle(letters)
  order.extend(rng.choice(groups[l]) for l in letters)
 optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr);logs=[];sample_counts=collections.Counter();selected=[];started=time.perf_counter();checkpoints=set(int(x) for x in args.checkpoints.split(','));checkpoints.add(args.steps)
 for step,index in enumerate(order[:args.steps],1):
  model.train();row=tr[index];inp={k:v.to(args.device) for k,v in encoded[index].items()};logits=model(**inp).logits[0,-1]
  # Full-vocabulary next-token CE: the prompt contributes no target/loss.
  loss=F.cross_entropy(logits[None],torch.tensor([ids[row['letter']]],device=args.device));optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optim.step();sample_counts[row['letter']]+=1
  logs.append({'step':step,'loss':float(loss.detach()),'letter':row['letter'],'prompt_tokens':int(inp['input_ids'].shape[-1])})
  if step%20==0:print(f'seed={args.seed} step={step}/{args.steps} token_loss={float(loss.detach()):.5f}',flush=True)
  if step in checkpoints:
   model.eval();folder=out/f'step_{step}';folder.mkdir(exist_ok=True);model.save_pretrained(folder);tokenizer.save_pretrained(folder)
   forced=teacher_forced(model,tokenizer,va,args.device)
   policy=object.__new__(lab.LMPolicy);policy.torch=torch;policy.model=model;policy.tokenizer=tokenizer;policy.device=args.device;policy.ids=ids
   results=[lab.rollout(c,'lm',policy,args.seed) for c in val_cases];types=sorted(set(c['scenario_type'] for c in val_cases));macro=sum(sum(r['success'] for r in results if r['scenario_type']==t)/sum(r['scenario_type']==t for r in results) for t in types)/len(types)
   record={'step':step,'validation_successes':sum(r['success'] for r in results),'validation_n':len(results),'validation_macro_scenario_success':macro,'teacher_forced':forced,
           'sampled_action_counts_so_far':dict(sample_counts),'adapter_sha256':lab.sha(folder/'adapter_model.safetensors')}
   lab.dump(folder/'validation.json',record);lab.jsonl(folder/'validation_traces.jsonl',results);selected.append(record);print('VALIDATION',json.dumps({k:v for k,v in record.items() if k!='teacher_forced'}),flush=True)
 best=max(selected,key=lambda r:(r['validation_macro_scenario_success'],r['teacher_forced']['terminal_accuracy'],-r['step']))
 summary={'seed':args.seed,'steps':args.steps,'selected_step':best['step'],'selection_rule':'highest validation macro scenario success; tie: terminal next-action accuracy; tie: earlier step',
          'checkpoints':selected,'sampled_action_counts':dict(sample_counts),'train_examples':len(tr),'val_examples':len(va),'train_recipe_groups':len(set(r['recipe_group'] for r in tr)),
          'max_prompt_tokens':max(int(x['input_ids'].shape[-1]) for x in encoded),'prompt_truncation_count':0,'objective':'full-vocabulary CE on exactly one next-action token; no prompt targets',
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':time.perf_counter()-started,'logs':logs,'args':vars(args),
          'data_hashes':{'train':lab.sha(data/'policy_train.jsonl'),'val':lab.sha(data/'policy_val.jsonl')},'source_hashes':{'agent_v2.py':lab.sha('agent_v2.py'),'agent_facts_v2.py':lab.sha('agent_facts_v2.py'),'train_facts_v2.py':lab.sha(__file__)},
          'test_opened_for_selection':False,'scope':'balanced simulated public-state policy supervision; no real company incident or API invocation'}
 lab.dump(out/'training_summary.json',summary);print('SELECTED',args.seed,best['step'],flush=True)
if __name__=='__main__':main()
