import os
for _threads in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
 os.environ.setdefault(_threads,"1")
import sqlite3,pathlib,json,numpy as np,time,math,itertools,hashlib,datetime,platform,argparse
_ap=argparse.ArgumentParser();_ap.add_argument('--data',type=pathlib.Path,required=True);_ap.add_argument('--out',type=pathlib.Path,required=True);_A=_ap.parse_args();data=_A.data;out=_A.out;con=sqlite3.connect(out/'baseline.sqlite');md=np.load(data/'submission_metadata.npz',allow_pickle=False);rec=md['mbid'].astype(str);build=md['build'].astype(str);digest=md['md5'].astype(str)
r=json.loads((out/'baseline_receipt.json').read_text());features=r['feature_queries'];cols=['loudness','complexity','bpm','centroid'];threshold={x['feature']:x['threshold'] for x in r['variable_membership_avg']};cases=[];sql_log=[]
def rows(recording):
 raw=con.execute('SELECT rowid-1,build,loudness,complexity,bpm,centroid FROM submission WHERE recording=? ORDER BY rowid',(recording,)).fetchall();unique={tuple(x[2:]):x for x in raw};return list(unique.values())
for j,(c,f) in enumerate(zip(cols,features)):
 t=threshold[f];q=f'''SELECT recording FROM submission WHERE split='development' GROUP BY recording HAVING MIN({c})<? AND MAX({c})>=? ORDER BY recording LIMIT100'''.replace('LIMIT100','LIMIT 100');start=time.perf_counter();candidates=con.execute(q,[t,t]).fetchall();scan_s=time.perf_counter()-start;sql_log.append({'sql':q,'parameters':[t,t],'elapsed_s':scan_s})
 cross=next(((rr,rows(rr)) for (rr,) in candidates if 2<=len(rows(rr))<=3),None)
 if cross is None:cases.append({'feature':f,'status':'no2-3candidatecrossingrecordintheselected100','searched_population':'development rows, first100IDs with a threshold crossing'});continue
 q=f'''SELECT recording FROM submission WHERE split='development' GROUP BY recording HAVING MIN({c})>=? ORDER BY recording LIMIT 100''';start=time.perf_counter();anchorc=con.execute(q,[t]).fetchall();scan2_s=time.perf_counter()-start;sql_log.append({'sql':q,'parameters':[t],'elapsed_s':scan2_s});anchor=next(((rr,rows(rr)) for (rr,) in anchorc if len(rows(rr))==1 and rr!=cross[0]),None);assert anchor
 choice=[cross[1],anchor[1]];answer=[];numeratorcount=[];worlds=[]
 for w in itertools.product(*choice):
  vals=[a[2+j] for a in w if a[2+j]>=t];num=sum(vals);count=len(vals);avg=num/count;answer.append(avg);numeratorcount.append([num,count]);worlds.append({'source_rows':[a[0] for a in w],'query_values':vals,'sum':num,'count':count,'avg':avg})
 
 base=anchor[1][0][2+j];passing=[a[2+j] for a in cross[1] if a[2+j]>=t];formula=[min([base,*[(base+v)/2 for v in passing]]),max([base,*[(base+v)/2 for v in passing]])];exact=[min(answer),max(answer)];assert max(abs(a-b) for a,b in zip(formula,exact))<1e-10;assert len(set(x[1] for x in numeratorcount))==2
 rectangle=[min(n for n,c in numeratorcount)/max(c for n,c in numeratorcount),max(n for n,c in numeratorcount)/min(c for n,c in numeratorcount)]
 cases.append({'feature':f,'threshold':t,'status':'real_two_recording_variable_membership_witness','recordings':[{'recording':rr,'candidates':[{'source_row_index':a[0],'build':a[1],'reported_encoded_md5':str(digest[a[0]]),'feature_values':list(a[2:])} for a in rs]} for rr,rs in [cross,anchor]],'worlds':worlds,'world_count':len(worlds),'exact_numeric_bounds':exact,'simple_optional_prefix_bounds':formula,'separate_numerator_count_box':rectangle,'box_width':rectangle[1]-rectangle[0],'exact_width':exact[1]-exact[0],'correctness_passed':True,'finding':'Actual denominator variation exists but an elementary matched-world baseline solves this witness. It is no new method or acoustic truth.'})

ratio=[]
for case in cases:
 if 'recordings' not in case:continue
 choice=[r['candidates'] for r in case['recordings']];values=[]
 for w in itertools.product(*choice):
  bpm=sum(x['feature_values'][2] for x in w);centroid=sum(x['feature_values'][3] for x in w);assert centroid>0;values.append(bpm/centroid)
 ratio.append({'from_case_feature':case['feature'],'query':'SUM(bpm)/SUM(spectral_centroid), preserving complete row choices','world_count':len(values),'exact_bounds':[min(values),max(values)],'finding':'Finite correlated-row enumeration exactly computes this small ratio; no claimed speed or theoretical separation.'})
outobj={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version(),'scope':'Targeted actual development candidate rows selected by a frozen full-development median threshold, not reported audio ground truth','sql_access_log':sql_log,'cases':cases,'ratio_queries':ratio,'new_algorithm_admitted':False}
(out/'targeted_variable_membership.json').write_text(json.dumps(outobj,ensure_ascii=False,indent=2));con.close();print(json.dumps({'witnesses':sum('worlds' in x for x in cases),'world_counts':[x.get('world_count') for x in cases],'ratio_queries':len(ratio),'correctness':[x.get('correctness_passed') for x in cases]},ensure_ascii=False))
