from pathlib import Path
import os
for variable in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
 os.environ[variable]="1"
import hashlib, itertools, json, sqlite3, time
import numpy as np
ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/"data"
RASTER=INPUT/"raster_panel.npz"
SEED=20261011

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def dump(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stable_hash(x):return int(hashlib.sha256((bytes.fromhex('4d555349435f5349474d4f445f76317c').decode('ascii')+str(x)).encode()).hexdigest()[:8],16)

def music():
 folder=INPUT/'audio'
 z=np.load(folder/'submission_metadata.npz',allow_pickle=False)
 x=np.asarray(np.load(folder/'submission_features.npy',mmap_mode='r'),np.float64)
 ids,group=np.unique(z['mbid'],return_inverse=True);count=np.bincount(group)
 mean=np.zeros((len(ids),4));np.add.at(mean,group,x);mean/=count[:,None]
 order=np.argsort(group,kind='stable');starts=np.r_[0,np.cumsum(count)]
 candidates=[np.unique(x[order[starts[i]:starts[i+1]]],axis=0) for i in range(len(ids))]
 hold=np.array([stable_hash(i)%5==0 for i in ids])
 return ids,candidates,mean,count,hold,z,x,group

def raster():
 z=np.load(RASTER,allow_pickle=False);a=z['integer_sums'][:,1].astype(np.float64)
 y=np.log1p(a);t=np.arange(16);centered=t-t.mean();post=np.arange(5)-2
 features=np.stack([y.mean(2),np.einsum('eci,i->ec',y,centered)/sum(centered**2),np.einsum('eci,i->ec',y[:,:,11:],post)/sum(post**2),y[:,:,11]-y[:,:,10]],axis=2)
 ids=np.array(['cell_'+str(i) for i in range(a.shape[1])]);candidates=[np.unique(features[:,i,:],axis=0) for i in range(len(ids))]
 mean=features.mean(0);count=np.full(len(ids),2);hold=np.array([stable_hash(i)%5==0 for i in ids])
 return ids,candidates,mean,count,hold,features

def world_answers(choices,mandatory,thresholds):
 n=len(choices);states=np.array(list(itertools.product((0,1),repeat=n)),np.uint8)
 values=np.array(choices)[np.arange(n)[None,:],states]
 qualifying=values[:,:,1]>=thresholds[1]
 averages=(np.sum(values[:,:,1]*qualifying,axis=1)+mandatory)/(qualifying.sum(1)+1)
 numeric=np.stack([values[:,:,0].sum(1),(values[:,:,2]>=thresholds[2]).sum(1),averages],axis=1)
 order=np.argsort(-values[:,:,3],axis=1,kind='stable');bits=np.left_shift(np.uint64(1),order.astype(np.uint64))
 sets=np.stack([np.bitwise_or.reduce(bits[:,:min(k,n)],axis=1) for k in (1,3)],axis=1)
 return states,values,numeric,sets

def scores(states,numeric,sets):
 n=states.shape[1];pop=np.array([i.bit_count() for i in range(1<<n)])
 scales=np.r_[numeric.max(0)-numeric.min(0),pop[np.bitwise_or.reduce(sets,axis=0)^np.bitwise_and.reduce(sets,axis=0)]]
 scales=np.where(scales>0,scales,1)
 results=np.zeros((1<<n,6))
 branch_records={}
 for mask in range(1<<n):
  reveal=[i for i in range(n) if mask>>i&1];b=len(reveal)
  group=states[:,reveal]@np.left_shift(1,np.arange(b)) if b else np.zeros(len(states),int)
  order=np.argsort(group,kind='stable');num=numeric[order].reshape(1<<b,-1,3);membership=sets[order].reshape(1<<b,-1,2)
  bounds=np.c_[num.max(1)-num.min(1),pop[np.bitwise_or.reduce(membership,axis=1)^np.bitwise_and.reduce(membership,axis=1)]]/scales
  results[mask,:5]=bounds.max(0);results[mask,5]=bounds.sum(1).max()
  branch_records[mask]=(group,bounds)
 return scales,results,branch_records

def adaptive(states,numeric,sets,scales,maxbudget=3):
 n=states.shape[1];pop=np.array([i.bit_count() for i in range(1<<n)]);memo={};loss={}
 def value(observed,budget):
  key=(observed,budget)
  if key in memo:return memo[key]
  if observed not in loss:
   indices=np.flatnonzero(np.all([(states[:,i]==v) for i,v in enumerate(observed) if v>=0],axis=0)) if any(v>=0 for v in observed) else np.arange(len(states))
   nv=numeric[indices];sv=sets[indices]
   loss[observed]=float(np.r_[nv.max(0)-nv.min(0),pop[np.bitwise_or.reduce(sv,axis=0)^np.bitwise_and.reduce(sv,axis=0)]]@ (1/scales))
  best=loss[observed]
  if budget:
   for i,v in enumerate(observed):
    if v>=0:continue
    children=[]
    for outcome in (0,1):
     child=list(observed);child[i]=outcome;children.append(value(tuple(child),budget-1))
    best=min(best,max(children))
  memo[key]=best;return best
 return {'budgets':[value((-1,)*n,b) for b in range(maxbudget+1)],'dynamic_program_states':len(memo),'compatible_state_losses':len(loss)}

def sql_check(states,values,numeric,sets,mandatory,thresholds):
 con=sqlite3.connect(':memory:');con.execute('CREATE TABLE candidate(world INT, entity INT,a REAL,b REAL,c REAL,d REAL)')
 con.executemany('INSERT INTO candidate VALUES (?,?,?,?,?,?)',((w,i,*v) for w,world in enumerate(values) for i,v in enumerate(world)))
 checks=[]
 queries=[('SELECT world,SUM(a) FROM candidate GROUP BY world',()),('SELECT world,SUM(CASE WHEN c>=? THEN 1 ELSE 0 END) FROM candidate GROUP BY world',(float(thresholds[2]),)),('SELECT world,(SUM(CASE WHEN b>=? THEN b ELSE 0 END)+?)/(SUM(CASE WHEN b>=? THEN 1 ELSE 0 END)+1.0) FROM candidate GROUP BY world',(float(thresholds[1]),float(mandatory),float(thresholds[1])))]
 for j,(query,args) in enumerate(queries):
  actual=np.array([v for w,v in con.execute(query,args)]);assert np.allclose(actual,numeric[:,j],rtol=1e-12,atol=1e-10);checks.append({'query':query,'world_answers':len(actual),'max_absolute_error':float(np.max(abs(actual-numeric[:,j])))})
 for k,j in [(1,0),(3,1)]:
  actual=np.zeros(len(states),np.uint64)
  query='WITH r AS (SELECT world,entity,ROW_NUMBER() OVER (PARTITION BY world ORDER BY d DESC,entity) pos FROM candidate) SELECT world,entity FROM r WHERE pos<=?'
  for w,i in con.execute(query,(min(k,states.shape[1]),)):actual[w]|=np.uint64(1)<<np.uint64(i)
  assert np.array_equal(actual,sets[:,j]);checks.append({'query':query,'k':k,'world_answers':len(actual),'sets_exact':True})
 con.close();return checks

def m2(out):
 rng=np.random.default_rng(SEED);results=[];budgets=[];world_checks=[]
 for domain,data in [('audio',music()),('raster',raster())]:
  ids,candidates,ref,count,hold=data[:5];thresholds=np.median(ref[~hold],axis=0)
  eligible=np.array([i for i,c in enumerate(candidates) if len(c)>1 and hold[i]])
  mandatory=float(ref[(~hold)&(ref[:,1]>=thresholds[1]),1].max())
  for n in (4,6,8):
   for case in range(100):
    selected=np.sort(rng.choice(eligible,n,replace=False));choices=[np.array([candidates[i][0],candidates[i][-1]]) for i in selected]
    states,values,num,sets=world_answers(choices,mandatory,thresholds)
    start=time.perf_counter();scales,score,branches=scores(states,num,sets);dt=time.perf_counter()-start
    masks=np.arange(1<<n);sizes=np.array([i.bit_count() for i in masks]);greedy=0;idmask=0;randommask=0;random_order=rng.permutation(n);width_order=np.argsort(-np.linalg.norm(np.array(choices)[:,1]-np.array(choices)[:,0],axis=1))
    widthmask=0;querymasks=[0]*5;case_rows=[]
    for budget in range(n+1):
     best=float(score[sizes==budget,5].min())
     if budget:
      greedy=min((greedy|(1<<i) for i in range(n) if not greedy>>i&1),key=lambda s:(score[s,5],s))
      idmask|=1<<(budget-1);randommask|=1<<int(random_order[budget-1]);widthmask|=1<<int(width_order[budget-1])
      for j in range(5):querymasks[j]=min((querymasks[j]|(1<<i) for i in range(n) if not querymasks[j]>>i&1),key=lambda s:(score[s,j],s))
     row={'domain':domain,'n':n,'case':case,'budget':budget,'batch_optimum':best,'joint_greedy':float(score[greedy,5]),'id_order':float(score[idmask,5]),'random_order':float(score[randommask,5]),'raw_width_order':float(score[widthmask,5]),'single_query_greedy':[float(score[s,5]) for s in querymasks],'optimal_masks':masks[(sizes==budget)&np.isclose(score[:,5],best,rtol=0,atol=1e-12)].tolist(),'joint_greedy_mask':int(greedy)}
     assert row['joint_greedy']+1e-10>=best;case_rows.append(row);budgets.append(row)
    adaptive_result=adaptive(states,num,sets,scales) if case<5 else None
    if adaptive_result:
     for b,loss in enumerate(adaptive_result['budgets']):assert loss<=case_rows[b]['batch_optimum']+1e-9
    checks=sql_check(states,values,num,sets,mandatory,thresholds)
    world_checks.append({'domain':domain,'n':n,'case':case,'checks':checks})
    result={'domain':domain,'n':n,'case':case,'ids':ids[selected].tolist(),'choices':[c.tolist() for c in choices],'fixed_qualifying_reference_value':mandatory,'development_thresholds':thresholds.tolist(),'worlds':len(states),'initial_query_scales':scales.tolist(),'joint_score_initial':float(score[0,5]),'score_all_subsets_seconds':dt,'greedy_gap_max':max(r['joint_greedy']-r['batch_optimum'] for r in case_rows),'adaptive':adaptive_result,'oracle':'abstract binary candidate-index revelation, without acoustic or human truth'}
    results.append(result)
    if case==0 or case==99:print(json.dumps({'stage':'query_verification','domain':domain,'n':n,'case':case,'greedy_gap':result['greedy_gap_max']}),flush=True)
    if case==0 or result['greedy_gap_max']>1e-6:
     np.savez_compressed(out/f'{domain}_n{n}_case{case}_worlds.npz',states=states,values=values,numeric=num,memberships=sets,scores=score)
 dump(out/'query_verification_all_cases.json',results);dump(out/'query_verification_all_budgets.json',budgets);dump(out/'query_verification_SQL_world_checks.json',world_checks)
 summary=[]
 for domain in ('audio','raster'):
  for n in (4,6,8):
   cases=[c for c in results if c['domain']==domain and c['n']==n];b=[r for r in budgets if r['domain']==domain and r['n']==n]
   summary.append({'domain':domain,'n':n,'cases':len(cases),'world_answers':sum(c['worlds']*5 for c in cases),'greedy_failure_cases':sum(c['greedy_gap_max']>1e-9 for c in cases),'max_greedy_gap':max(c['greedy_gap_max'] for c in cases),'adaptive_cases':sum(c['adaptive'] is not None for c in cases),'adaptive_better_than_batch_cases':sum(c['adaptive'] is not None and any(c['adaptive']['budgets'][i]<next(x['batch_optimum'] for x in b if x['case']==c['case'] and x['budget']==i)-1e-9 for i in range(4)) for c in cases),'median_exact_subset_score_seconds':float(np.median([c['score_all_subsets_seconds'] for c in cases]))})
 dump(out/'query_verification_summary.json',summary);return summary
