#!/usr/bin/env python3
"""Teacher-forced public-prefix accuracy; separate from closed-loop task success."""
import argparse,json
from pathlib import Path
import agent_v2 as lab

def main():
 p=argparse.ArgumentParser();p.add_argument('--rows',default='data/policy_val.jsonl');p.add_argument('--model',default=lab.MODEL_ID);p.add_argument('--adapter');p.add_argument('--out',required=True);p.add_argument('--representation',choices=['raw','facts'],default='raw');p.add_argument('--terminal-only',action='store_true');p.add_argument('--one-per-scenario',action='store_true');args=p.parse_args()
 rows=lab.load_jsonl(args.rows)
 if args.terminal_only:rows=[r for r in rows if lab.ACTIONS[r['letter']][0] in ['answer','handoff']]
 if args.one_per_scenario:
  found={}
  for r in rows:found.setdefault(r['scenario_type'],r)
  rows=list(found.values())
 if args.representation=='facts':
  from agent_facts_v2 import LMPolicy
 else:LMPolicy=lab.LMPolicy
 lm=LMPolicy(args.model,args.adapter,threads=1);results=[]
 for r in rows:
  pred,usage=lm.choose(r['state']);results.append({'scenario_type':r['scenario_type'],'recipe_group':r['recipe_group'],'true_letter':r['letter'],'predicted_letter':pred,'correct':pred==r['letter'],'usage':usage})
 s={'examples':len(results),'correct':sum(r['correct'] for r in results),'accuracy':sum(r['correct'] for r in results)/len(results),'args':vars(args),'scope':'Teacher-forced public prefixes, not closed-loop task success'}
 lab.dump(args.out,{'summary':s,'rows':results});print(json.dumps(s))
if __name__=='__main__':main()
