import os
for _threads in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
 os.environ.setdefault(_threads,"1")
import pathlib,json,sqlite3,numpy as np,hashlib,time,datetime,platform,resource,itertools,math,argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--data',type=pathlib.Path,required=True);_ap.add_argument('--results',type=pathlib.Path,required=True);_ap.add_argument('--code',type=pathlib.Path,required=True);_ap.add_argument('--out',type=pathlib.Path,required=True);_A=_ap.parse_args();data=_A.data;results=_A.results;code=_A.code;out=_A.out;out.mkdir(parents=True,exist_ok=True)
T=time.perf_counter();timing={};t=time.perf_counter();md=np.load(data/'submission_metadata.npz',allow_pickle=False);rec=md['mbid'].astype(str);build=md['build'].astype(str);digest=md['md5'].astype(str);features=['lowlevel.average_loudness','lowlevel.dynamic_complexity','rhythm.bpm','lowlevel.spectral_centroid.mean'];fi=[np.flatnonzero(md['feature'].astype(str)==x).item() for x in features];timing['metadata_load_s']=time.perf_counter()-t
t=time.perf_counter();X=np.load(data/'submission_features.npy',mmap_mode='r');V=np.asarray(X[:,fi],dtype=float);assert np.isfinite(V).all();timing['feature_projection_load_s']=time.perf_counter()-t
reference=json.loads((results/'uncertainty_queries.json').read_text());builds=reference[0]['builds'];assert all(r['builds']==builds for r in reference)
t=time.perf_counter();split=['holdout' if int(hashlib.sha256((bytes.fromhex('4d555349435f5349474d4f445f76317c').decode('ascii')+x).encode()).hexdigest()[:8],16)%5==0 else 'development' for x in rec];timing['split_reconstruction_s']=time.perf_counter()-t
con=sqlite3.connect(out/'baseline.sqlite');con.execute('PRAGMA journal_mode=DELETE');con.execute('PRAGMA synchronous=FULL');con.execute('DROP TABLE IF EXISTS submission');con.execute('CREATE TABLE submission (recording TEXT NOT NULL, build TEXT NOT NULL, split TEXT NOT NULL, loudness REAL NOT NULL, complexity REAL NOT NULL, bpm REAL NOT NULL, centroid REAL NOT NULL)')
t=time.perf_counter();con.executemany('INSERT INTO submission VALUES (?,?,?,?,?,?,?)',((str(rec[i]),str(build[i]),split[i],*[float(x) for x in V[i]]) for i in range(len(rec))));con.commit();timing['sqlite_insert_s']=time.perf_counter()-t
t=time.perf_counter();con.execute('CREATE INDEX submission_build_split_recording ON submission (build,split,recording)');con.commit();timing['sqlite_index_s']=time.perf_counter()-t
cols=['loudness','complexity','bpm','centroid'];queries=[];checks=[]
def sql_for(c):return f'''WITH rb AS MATERIALIZED (
 SELECT split,build,recording,AVG({c}) AS v FROM submission
 WHERE build IN (?,?) GROUP BY split,build,recording
), common AS (
 SELECT split,recording FROM rb GROUP BY split,recording HAVING COUNT(DISTINCT build)=2
), totals AS (
 SELECT r.split,r.build,COUNT(*) AS n_recordings,SUM(r.v) AS world_sum
 FROM rb AS r JOIN common AS c USING(split,recording)
 GROUP BY r.split,r.build
)
SELECT split,MIN(n_recordings),MIN(world_sum),MAX(world_sum) FROM totals GROUP BY split ORDER BY split;'''
for c,f in zip(cols,features):
 q=sql_for(c);queries.append({'id':'shared_build_sum_'+c,'sql':q,'parameters':builds})
 elapsed=[];rows=None
 for rep in range(3):
  t=time.perf_counter();rows=con.execute(q,builds).fetchall();elapsed.append(time.perf_counter()-t)
 for s,n,lo,hi in rows:
  target=next(x for x in reference if x['feature']==f and x['split']==s);assert n==target['common_recordings'];delta=max(abs(lo-target['shared_build_sum_bounds'][0]),abs(hi-target['shared_build_sum_bounds'][1]));assert delta<1e-8
  checks.append({'query':'shared_build_sum','feature':f,'split':s,'recordings':n,'sqlite_bounds':[lo,hi],'root_bounds':target['shared_build_sum_bounds'],'max_absolute_difference':delta,'elapsed_query_only_s':elapsed,'meaning':'Same globally shared build and same recording-within-build means as root; independent interval widths not used as method-quality baseline.'})

policy=np.load(results/'policy_recordings.npz',allow_pickle=False);dev=~policy['holdout'];thresholds=[float(np.quantile(policy['recording_allmean'][dev,j],.5)) for j in range(4)]
variable=[];topk=[]
for c,f,threshold in zip(cols,features,thresholds):
 q=f'''WITH rb AS MATERIALIZED (
 SELECT split,build,recording,AVG({c}) AS v FROM submission
 WHERE build IN (?,?) GROUP BY split,build,recording
), common AS (SELECT split,recording FROM rb GROUP BY split,recording HAVING COUNT(DISTINCT build)=2)
SELECT r.split,r.build,COUNT(*) AS denominator,SUM(r.v) AS numerator,AVG(r.v) AS answer
FROM rb r JOIN common c USING(split,recording) WHERE r.v>=?
GROUP BY r.split,r.build ORDER BY r.split,r.build;'''
 params=[*builds,threshold];t=time.perf_counter();ans=con.execute(q,params).fetchall();dt=time.perf_counter()-t;queries.append({'id':'predicate_avg_'+c,'sql':q,'parameters':params})
 
 grouped={}
 for i in range(len(rec)):
  if build[i] in builds:
   grouped.setdefault((split[i],str(build[i]),str(rec[i])),[]).append(float(V[i,cols.index(c)]))
 for s in ['development','holdout']:
  sets=[{r for ss,b,r in grouped if ss==s and b==bb} for bb in builds];common=set.intersection(*sets);worlds=[]
  for bb in builds:
   vals=[float(np.mean(grouped[(s,bb,r)])) for r in sorted(common)];passed=[v for v in vals if v>=threshold];n=len(passed);num=sum(passed);avg=num/n if n else None
   got=next((x for x in ans if x[0]==s and x[1]==bb),None)
   if n:
    assert got and got[2]==n and abs(got[3]-num)<1e-8 and abs(got[4]-avg)<1e-8
   else:assert got is None
   worlds.append({'build':bb,'denominator':n,'numerator':num,'avg':avg})
  valid=[w['avg'] for w in worlds if w['avg'] is not None];variable.append({'feature':f,'split':s,'threshold':threshold,'worlds':worlds,'numeric_nonempty_bounds':[min(valid),max(valid)] if valid else None,'empty_world_possible':any(w['denominator']==0 for w in worlds),'query_only_s':dt,'matched_array_check':True})
 q=f'''WITH rb AS MATERIALIZED (
 SELECT split,build,recording,AVG({c}) AS v FROM submission
 WHERE build IN (?,?) GROUP BY split,build,recording
), common AS (SELECT split,recording FROM rb GROUP BY split,recording HAVING COUNT(DISTINCT build)=2), ranked AS (
 SELECT r.split,r.build,r.recording,r.v,ROW_NUMBER() OVER(PARTITION BY r.split,r.build ORDER BY r.v DESC,r.recording) AS pos
 FROM rb r JOIN common c USING(split,recording)
)
SELECT split,build,recording,v,pos FROM ranked WHERE pos<=5 ORDER BY split,build,pos;'''
 t=time.perf_counter();ans=con.execute(q,builds).fetchall();dt=time.perf_counter()-t;queries.append({'id':'top5_'+c,'sql':q,'parameters':builds})
 for s in ['development','holdout']:
  worldsets=[]
  for bb in builds:
   got=[r[2] for r in ans if r[0]==s and r[1]==bb];common=set.intersection(*[{r for ss,b,r in grouped if ss==s and b==bc} for bc in builds]);expected=sorted(common,key=lambda r:(-np.mean(grouped[(s,bb,r)]),r))[:5];assert got==expected;worldsets.append(set(got))
  topk.append({'feature':f,'split':s,'k':5,'worlds':[{'build':bb,'top_ids':[r[2] for r in ans if r[0]==s and r[1]==bb]} for bb in builds],'certain_ids':sorted(set.intersection(*worldsets)),'possible_ids':sorted(set.union(*worldsets)),'query_only_s':dt,'matched_array_check':True})

byrec={}
for i,r in enumerate(rec):
 if split[i]=='development':byrec.setdefault(str(r),[]).append(i)
selected=[]
for r,ix in sorted(byrec.items()):
 unique={tuple(float(x) for x in V[i]):i for i in ix}
 if 2<=len(unique)<=3: selected.append((r,list(unique.values())))
 if len(selected)==6:break
assert len(selected)==6
choices=[[tuple(float(x) for x in V[i]) for i in ix] for _,ix in selected];combos=math.prod(map(len,choices));assert combos<=729
small=[]
def independent_optional_mean_bounds(cs,j,t):
 forced_min=[];forced_max=[];optional_min=[];optional_max=[]
 for values in cs:
  yes=[x[j] for x in values if x[j]>=t];no=[x[j] for x in values if x[j]<t]
  if not yes:continue
  if no:optional_min.append(min(yes));optional_max.append(max(yes))
  else:forced_min.append(min(yes));forced_max.append(max(yes))
 def prefixes(base,opt,reverse):
  total=sum(base);n=len(base);scores=[total/n] if n else []
  for v in sorted(opt,reverse=reverse):total+=v;n+=1;scores.append(total/n)
  return scores
 lo=prefixes(forced_min,optional_min,False);hi=prefixes(forced_max,optional_max,True)
 return ([min(lo),max(hi)] if lo else None,not forced_min)
for j,f in enumerate(features):
 
 t=thresholds[j];answers=[];empty=0;t0=time.perf_counter()
 for world in itertools.product(*choices):
  selected_vals=[v[j] for v in world if v[j]>=t]
  if not selected_vals:empty+=1
  else:answers.append(sum(selected_vals)/len(selected_vals))
 enum_s=time.perf_counter()-t0;t0=time.perf_counter();formula,empty_possible=independent_optional_mean_bounds(choices,j,t);formula_s=time.perf_counter()-t0;exact=[min(answers),max(answers)] if answers else None
 if exact:assert max(abs(a-b) for a,b in zip(exact,formula))<1e-10
 else:assert formula is None
 assert empty_possible==(empty>0)
 small.append({'feature':f,'threshold':t,'worlds_enumerated':combos,'numeric_nonempty_exact':exact,'optional_prefix_baseline':formula,'empty_world_count':empty,'empty_possible_baseline':empty_possible,'enumeration_query_only_s':enum_s,'prefix_query_only_s':formula_s,'correctness_passed':True})

smallrows=[{'recording':r,'candidates':[{'source_row_index':i,'build':str(build[i]),'reported_encoded_md5':str(digest[i]),'feature_values':[float(x) for x in V[i]]} for i in ix]} for r,ix in selected]
(out/'independent_actual_candidates.json').write_text(json.dumps({'features':features,'records':smallrows,'selection':'first6developmentIDs with2-3distinct full four-field row payloads; all qualifying candidate vectors retained for each selected recording','oracle':'none; only observed candidate-world semantics'},ensure_ascii=False,indent=2));(out/'queries.sql').write_text('\n\n'.join('-- '+q['id']+'\n'+q['sql'] for q in queries));(out/'queries.json').write_text(json.dumps(queries,ensure_ascii=False,indent=2))
receipt={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'root_code_sha256':hashlib.sha256((code/'run_queries.py').read_bytes()).hexdigest(),'root_results_sha256':{n:hashlib.sha256((results/n).read_bytes()).hexdigest() for n in ['uncertainty_queries.json','matched_payload_build.json','verification_queries.json','exhaustive_correctness.json']},'source_receipt_sha256':hashlib.sha256((data/'intake_receipt.json').read_bytes()).hexdigest(),'source_data_hashes_from_intake_receipt':json.loads((data/'intake_receipt.json').read_text())['outputs'],'software':{'python':platform.python_version(),'numpy':np.__version__,'sqlite':sqlite3.sqlite_version},'submitted_rows':len(rec),'feature_queries':features,'builds':builds,'timing':timing,'fixed_sum_checks':checks,'variable_membership_avg':variable,'topk_checks':topk,'independent_small_worlds':small,'total_elapsed_s':time.perf_counter()-T,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'all_checks_passed':True,'admission':{'fixed_shared_build_sum':'Ordinary group-by-build SQL exactly solves the evaluated queries','global_build_where_avg_topk':'The same finite two-world SQL evaluation exactly solves the evaluated queries','independent_predicate_avg':'small actual exhaustive match to elementary optional-prefix baseline; no novel algorithm claimed','current_verification_curves':'independent-world width removal; do not compare against the distinct shared-build world contract','audio_truth':'not established','new_algorithm_admitted':False}}
(out/'baseline_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));con.close();print(json.dumps({'sum_checks':len(checks),'variable_avg_checks':len(variable),'topk_checks':len(topk),'small_checks':len(small),'actual_small_worlds':combos,'all_checks_passed':True,'timing':timing,'total_elapsed_s':receipt['total_elapsed_s'],'median_sum_query_s':[float(np.median(x['elapsed_query_only_s'])) for x in checks]},ensure_ascii=False))
