import numpy as np
from . import experiment as E

def run(O):
 def dump(name,x):E.dump(O/name,x)
 ids,cands,ref,count,hold,features=E.raster();thresholds=np.median(ref[~hold],axis=0);mandatory=float(ref[(~hold)&(ref[:,1]>=thresholds[1]),1].max());rng=np.random.default_rng(20261012);eligible=np.flatnonzero(hold & np.array([len(c)>1 for c in cands]));out=[]
 for n in (4,6,8):
  for case in range(100):
   sel=np.sort(rng.choice(eligible,n,replace=False));choice=np.moveaxis(features[:,sel,:],0,1);states,values,num,sets=E.world_answers(choice,mandatory,thresholds);endpoints=np.flatnonzero(np.all(states==0,axis=1)|np.all(states==1,axis=1));global_num=num[endpoints];global_sets=sets[endpoints]
   widths=num.max(0)-num.min(0);gw=global_num.max(0)-global_num.min(0);assert np.all(gw<=widths+1e-9)
   assert all(not np.array_equal(features[0,i],features[1,i]) for i in sel)
   pop=np.array([i.bit_count() for i in range(1<<n)]);pm=pop[np.bitwise_or.reduce(sets,axis=0)^np.bitwise_and.reduce(sets,axis=0)];gp=pop[np.bitwise_or.reduce(global_sets,axis=0)^np.bitwise_and.reduce(global_sets,axis=0)]
   assert np.all(gp<=pm)
   out.append({'n':n,'case':case,'ids':ids[sel].tolist(),'independent_worlds':len(states),'global_worlds':2,'numeric_independent_widths':widths.tolist(),'numeric_global_widths':gw.tolist(),'independent_membership_widths':pm.tolist(),'global_membership_widths':gp.tolist(),'one_global_reveal_residual_uncertainty':0})
 dump('query_verification_global_edition_controls.json',out)
 summary=[]
 for n in (4,6,8):
  a=[r for r in out if r['n']==n];summary.append({'n':n,'cases':len(a),'sum_strictly_wider':sum(r['numeric_independent_widths'][0]>r['numeric_global_widths'][0]+1e-9 for r in a),'count_strictly_wider':sum(r['numeric_independent_widths'][1]>r['numeric_global_widths'][1]+1e-9 for r in a),'avg_strictly_wider':sum(r['numeric_independent_widths'][2]>r['numeric_global_widths'][2]+1e-9 for r in a),'top1_strictly_wider':sum(r['independent_membership_widths'][0]>r['global_membership_widths'][0] for r in a),'top3_strictly_wider':sum(r['independent_membership_widths'][1]>r['global_membership_widths'][1] for r in a)})
 dump('query_verification_global_edition_summary.json',summary)
