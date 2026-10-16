import pathlib,json,itertools,sqlite3,time,datetime,hashlib,math,platform,argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--source',type=pathlib.Path,required=True);_ap.add_argument('--out',type=pathlib.Path,required=True);_A=_ap.parse_args();source=_A.source;out=_A.out;out.mkdir(parents=True,exist_ok=True);cases=json.loads(source.read_text())['cases'];records={}
for case in cases:
 for r in case.get('recordings',[]):records[r['recording']]=r
records=[records[k] for k in sorted(records)];assert records;ids=[r['recording'] for r in records];choices=[r['candidates'] for r in records];states=list(itertools.product(*[range(len(c)) for c in choices]));assert len(states)<=256;rows=[[choices[i][v]['feature_values'] for i,v in enumerate(s)] for s in states];actions=[i for i,c in enumerate(choices) if len(c)>1];assert len(actions)<=4
threshold={c['feature']:c['threshold'] for c in cases};bpm_t=threshold['rhythm.bpm'];complex_t=threshold['lowlevel.dynamic_complexity'];Q=['count_distinct_fixed','count_distinct_bpm_filter','where_avg_complexity','top1_centroid','top3_centroid','top5_centroid'];answers=[]
for world in rows:
 selected=[x[1] for x in world if x[1]>=complex_t];assert selected
 order=sorted(range(len(world)),key=lambda i:(-world[i][3],ids[i]));answers.append([len(ids),sum(x[2]>=bpm_t for x in world),sum(selected)/len(selected),*[frozenset(ids[i] for i in order[:min(k,len(ids))]) for k in [1,3,5]]])
def uncertainty(indices,j):
 vals=[answers[i][j] for i in indices]
 if j<3:return max(vals)-min(vals)
 return len(set.union(*[set(x) for x in vals])-set.intersection(*[set(x) for x in vals]))
scales=[uncertainty(range(len(states)),j) for j in range(len(Q))]
def branches(subset):
 groups={}
 for w,s in enumerate(states):groups.setdefault(tuple(s[i] for i in subset),[]).append(w)
 return groups
cache={}
def score(subset,only=None):
 subset=tuple(sorted(subset));key=(subset,only)
 if key in cache:return cache[key]
 worst=0;per=[]
 for observed,indices in branches(subset).items():
  vec=[uncertainty(indices,j)/scales[j] if scales[j] else 0 for j in range(len(Q))];value=vec[only] if only is not None else sum(vec);worst=max(worst,value);per.append({'reveal_indices':list(observed),'compatible_worlds':len(indices),'normalized_query_uncertainties':vec,'joint_loss':sum(vec)})
 result={'worst_residual_loss':worst,'branches':per};cache[key]=result;return result
individual=[]
for j,q in enumerate(Q):
 if scales[j]==0:individual.append({'query':q,'initial_uncertainty':0,'one_action_choices':[],'finding':'Already certain; zero reveal is sufficient.'});continue
 best=min(score([i],j)['worst_residual_loss'] for i in actions);wins=[i for i in actions if abs(score([i],j)['worst_residual_loss']-best)<1e-12];individual.append({'query':q,'initial_uncertainty':scales[j],'one_action_choices':[ids[i] for i in wins],'minimum_normalized_worst_loss':best})
results=[]
for budget in range(len(actions)+1):
 possible=list(itertools.combinations(actions,budget));best=min(score(s)['worst_residual_loss'] for s in possible);optimal=[s for s in possible if abs(score(s)['worst_residual_loss']-best)<1e-12]
 joint=[]
 for b in range(budget):
  candidates=[i for i in actions if i not in joint];chosen=min(candidates,key=lambda i:(score([*joint,i])['worst_residual_loss'],ids[i]));joint.append(chosen)
 policies={'joint_greedy':joint,'id_order':actions[:budget]}
 for j,q in enumerate(Q):
  selected=[]
  for b in range(budget):
   cand=[i for i in actions if i not in selected];selected.append(min(cand,key=lambda i:(score([*selected,i],j)['worst_residual_loss'],ids[i])))
  policies['single_query_greedy_'+q]=selected
 results.append({'budget':budget,'brute_force_optimal_worst_loss':best,'optimal_action_sets':[[ids[i] for i in s] for s in optimal],'number_subsets_evaluated':len(possible),'policies':{name:{'records':[ids[i] for i in chosen],'joint_worst_loss':score(chosen)['worst_residual_loss'],'gap_to_static_optimal':score(chosen)['worst_residual_loss']-best} for name,chosen in policies.items()}})

con=sqlite3.connect(':memory:');con.execute('CREATE TABLE world_row (world INTEGER,recording TEXT,loudness REAL,complexity REAL,bpm REAL,centroid REAL)');con.executemany('INSERT INTO world_row VALUES (?,?,?,?,?,?)',((w,ids[i],*row) for w,world in enumerate(rows) for i,row in enumerate(world)));sql_queries=[('SELECT world,COUNT(DISTINCT recording) FROM world_row GROUP BY world',[]),('SELECT world,COUNT(DISTINCT recording) FROM world_row WHERE bpm>=? GROUP BY world',[bpm_t]),('SELECT world,AVG(complexity) FROM world_row WHERE complexity>=? GROUP BY world',[complex_t])];timings=[]
for j,(query,params) in enumerate(sql_queries):
 t=time.perf_counter();sqlrows=dict(con.execute(query,params).fetchall());dt=time.perf_counter()-t
 for w in range(len(states)):
  got=sqlrows.get(w,0);assert abs(got-answers[w][j])<1e-10
 timings.append({'query':Q[j],'sql':query,'parameters':params,'query_only_s':dt,'all_world_answers_match':True})
for k,j in zip([1,3,5],[3,4,5]):
 query='WITH ranked AS (SELECT world,recording,ROW_NUMBER() OVER (PARTITION BY world ORDER BY centroid DESC,recording) pos FROM world_row) SELECT world,recording FROM ranked WHERE pos<=?';t=time.perf_counter();ans=con.execute(query,[min(k,len(ids))]).fetchall();dt=time.perf_counter()-t;sqlsets={}
 for w,r in ans:sqlsets.setdefault(w,set()).add(r)
 for w in range(len(states)):assert sqlsets[w]==set(answers[w][j])
 timings.append({'query':Q[j],'sql':query,'parameters':[min(k,len(ids))],'query_only_s':dt,'all_world_answers_match':True})
conflicts=[]
for a,b in itertools.combinations([x for x in individual if x['initial_uncertainty']],2):
 if set(a['one_action_choices']).isdisjoint(b['one_action_choices']):conflicts.append({'query_pair':[a['query'],b['query']],'disjoint_best_actions':[a['one_action_choices'],b['one_action_choices']]})
truth=tuple(0 for c in choices);controlled=[{'recording':ids[i],'revealed_candidate_index':truth[i],'source_row_index':choices[i][truth[i]]['source_row_index']} for i in actions]
report={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'actual_sample_candidate_slice_with_constructed_query_workload','query_selection':'Exploratory joint workload, thresholds frozen at root development medians. All selected candidate rows come from the actual parsed sample; these queries and normalized equal-weight joint goal were added after examining the fixed-SUM failure.','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'records':records,'worlds':states,'world_count':len(states),'recording_count':len(ids),'uncertain_recordings':len(actions),'query_names':Q,'initial_uncertainty_scales':scales,'objective':'Static batch of equal abstract-cost candidate-index reveals; minimize worst compatible-world sum of initial-scale-normalized numeric widths or possible-minus-certain top-k memberships. No probabilities, no human cost, no adaptive-policy optimality claim.','individual_best_actions':individual,'selection_conflicts':conflicts,'budgets':results,'sql_checks':timings,'controlled_truth_scope':'Candidate-index zero for every selected recording; disclosed controlled choice, not acoustically validated truth','controlled_reveals':controlled,'all_sql_world_checks_passed':True,'new_algorithm_admitted':False,'interpretation':'Conflicting query-specific choices motivate declaring a joint workload loss. Exhaustive standard possible-world evaluation computes all scores and the brute-force static optimum for this slice; no algorithm separation has been demonstrated.'}
(out/'joint_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(out/'all_world_answers.json').write_text(json.dumps({'queries':Q,'answers':[[sorted(a) if isinstance(a,frozenset) else a for a in r] for r in answers]},ensure_ascii=False,indent=2));print(json.dumps({'records':len(ids),'uncertain_recordings':len(actions),'worlds':len(states),'query_conflicts':len(conflicts),'sql_checks':len(timings),'greedy_gaps':[r['policies']['joint_greedy']['gap_to_static_optimal'] for r in results]},ensure_ascii=False))
