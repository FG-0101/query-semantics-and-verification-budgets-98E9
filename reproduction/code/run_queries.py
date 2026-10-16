#!/usr/bin/env python3
"""Compute the fixed recording split, six archive policies and query outputs."""
import argparse, collections, hashlib, itertools, json, math, os, platform, resource, time
from pathlib import Path
os.environ.setdefault("OPENBLAS_NUM_THREADS","1");os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np

FEATURES=["lowlevel.average_loudness","lowlevel.dynamic_complexity","rhythm.bpm","lowlevel.spectral_centroid.mean"]
SEED=20261010
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
def grouped(values,inv,n):
    sums=np.zeros((n,values.shape[1]),np.float64);np.add.at(sums,inv,values)
    count=np.bincount(inv,minlength=n);return sums/np.maximum(count[:,None],1),count
def finite(v):return None if not np.isfinite(v) else float(v)
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    config={"feature_queries":FEATURES,"seed":SEED,
      "split":"SHA256(prefix + recording) first eight hex digits modulo 5, with ASCII prefix decoded from hex 4d555349435f5349474d4f445f76317c and zero assigned to holdout",
      "reference":"Equal recording weight, all observed submissions averaged within recording; descriptive reference, no acoustic ground truth",
      "date_groups":"Minimum parsed first reported tag year within recording. Unknown dates retained. A date group is metadata, not recording-session truth.",
      "policies":["all_submission","recording_allmean","recording_payloadmean","recording_fixed","recording_modal_payload","recording_dominant_build"],
      "build_selection":"Most frequent build among development submission rows; holdout cannot change selection",
      "ranking":"50 anchors per split, selected by ID hash; top 20 neighbors in four fixed scalar descriptors; scale learned on development recording means",
      "query_verification_semantics":"One globally shared extractor build for a fixed common-support set; independent per-record candidates are a wider model",
      "query_verification_audit":"Known-answer synthetic exhaustive worlds; archive verification is candidate revelation only, without human labels/cost",
      "bootstrap":"1000 recording-cluster paired resamples for fixed-vs-allmean contrasts; fixed query family, Holm correction is not used because no null-hypothesis testing is reported"}
    dump(a.out/"protocol.json",config)
    md=np.load(a.data/"submission_metadata.npz",allow_pickle=False)
    feat=md["feature"].astype(str);idx=[np.flatnonzero(feat==x).item() for x in FEATURES]
    X=np.load(a.data/"submission_features.npy",mmap_mode="r")
    V=np.asarray(X[:,idx],np.float64)
    rec=md["mbid"].astype(str);pay=md["md5"].astype(str);build=md["build"].astype(str);off=md["offset"]
    ids,ri=np.unique(rec,return_inverse=True);R=len(ids);allmean,nsub=grouped(V,ri,R)
    h=np.array([int(hashlib.sha256((bytes.fromhex('4d555349435f5349474d4f445f76317c').decode('ascii')+r).encode()).hexdigest()[:8],16) for r in ids],np.uint64)
    hold=h%5==0;split={"development":~hold,"holdout":hold}
    fixed=np.full(R,-1,np.int64)
    sort=np.lexsort((np.arange(len(rec)),build,pay,off,ri))
    begin=np.r_[True,ri[sort][1:]!=ri[sort][:-1]]
    fixed[ri[sort][begin]]=sort[begin]
    pidentifier=np.where(pay!="",pay,np.char.add("missing-row-",np.arange(len(pay)).astype(str)))
    pkeys=np.char.add(np.char.add(rec,"|"),pidentifier)
    pk,pi=np.unique(pkeys,return_inverse=True)
    pv,pcount=grouped(V,pi,len(pk))
    psort=np.argsort(pi,kind="stable");pb=np.r_[True,pi[psort][1:]!=pi[psort][:-1]]
    pfirst=psort[pb];pr=ri[pfirst]
    pmean,npay=grouped(pv,pr,R)
    modal=np.full(R,-1,np.int64)
    
    ms=np.lexsort((pk,-pcount,pr));mb=np.r_[True,pr[ms][1:]!=pr[ms][:-1]];modal[pr[ms][mb]]=ms[mb]
    devrows=(~hold)[ri]
    bu,bc=np.unique(build[devrows],return_counts=True);bo=np.lexsort((bu,-bc));dominant=bu[bo[0]]
    mask=build==dominant;bv,bn=grouped(V[mask],ri[mask],R);bv[bn==0]=np.nan
    pol={"recording_allmean":allmean,"recording_payloadmean":pmean,"recording_fixed":V[fixed],"recording_modal_payload":pv[modal],"recording_dominant_build":bv}
    years=np.full(R,np.iinfo(np.int32).max,np.int64);valid=np.zeros(len(rec),bool);dateyears=np.zeros(len(rec),np.int64)
    import re
    for i,s in enumerate(md["date"].astype(str)):
        m=re.match(r"^(\d{4})(?:\D|$)",s)
        if m:
            yy=int(m.group(1))
            if 1<=yy<=9999: valid[i]=True;dateyears[i]=yy
    np.minimum.at(years,ri[valid],dateyears[valid]);years[years==np.iinfo(np.int32).max]=0
    ysort=np.lexsort((dateyears,ri));pairs=set(zip(ri[valid].tolist(),dateyears[valid].tolist()))
    ny=np.bincount([r for r,y in pairs],minlength=R);decade=(years//10)*10
    composition={"rows":len(rec),"recordings":R,"distinct_payloads_within_recording":len(pk),"distinct_builds":len(set(build)),
      "multi_submission_recordings":int((nsub>1).sum()),"multi_payload_recordings":int((npay>1).sum()),
      "recordings_with_multiple_reported_years":int((ny>1).sum()),"unknown_tag_year_recordings":int((years==0).sum()),
      "selected_dominant_build":str(dominant),"dominant_build_recordings":int((bn>0).sum()),
      "recording_offset_duplicates":len(rec)-len(set(zip(rec.tolist(),off.tolist()))),
      "global_payloads_attached_to_multiple_recordings":int((np.bincount(np.unique(pay,return_inverse=True)[1][pfirst[pay[pfirst]!=""]])>1).sum()),
      "split_counts":{s:{"recordings":int(m.sum()),"submissions":int(m[ri].sum()),"dominant_build_covered_recordings":int((m&(bn>0)).sum())} for s,m in split.items()}}
    dump(a.out/"composition.json",composition)
    answers=[];identity=[];boot=[]
    rng=np.random.default_rng(SEED)
    for s,sm in split.items():
        group_masks=[("all",sm)]+[(str(int(d)),sm&(decade==d)) for d in np.unique(decade[sm])]
        for name,m in group_masks:
            if not m.any():continue
            rows=m[ri];ref=allmean[m].mean(0);sub=V[rows].mean(0)
            cov=((nsub[m,None]-nsub[m].mean())*(allmean[m]-ref)).mean(0)/nsub[m].mean()
            residual=sub-ref-cov
            identity.append({"split":s,"group":name,"n_recordings":int(m.sum()),"residual_max":float(np.max(np.abs(residual)))})
            for j,f in enumerate(FEATURES):
                answers.append({"split":s,"group":name,"feature":f,"policy":"all_submission","unit":"submission","denominator":int(rows.sum()),
                    "answer":float(sub[j]),"recording_mean_reference":float(ref[j]),"descriptive_difference":float(sub[j]-ref[j]),"fixed_support_difference":float(sub[j]-ref[j])})
            for pn,values in pol.items():
                support=m&np.isfinite(values).all(1)
                rref=allmean[support].mean(0) if support.any() else np.full(len(FEATURES),np.nan)
                ans=values[support].mean(0) if support.any() else np.full(len(FEATURES),np.nan)
                for j,f in enumerate(FEATURES):
                    answers.append({"split":s,"group":name,"feature":f,"policy":pn,"unit":"recording","denominator":int(support.sum()),
                      "answer":finite(ans[j]),"recording_mean_reference":float(ref[j]),"descriptive_difference":finite(ans[j]-ref[j]),
                      "fixed_support_difference":finite(ans[j]-rref[j])})
        delta=pol["recording_fixed"][sm]-allmean[sm]
        for j,f in enumerate(FEATURES):
            d=delta[:,j];B=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(1000)])
            boot.append({"split":s,"feature":f,"paired_recording_mean_difference":float(d.mean()),
                "percentile_95_interval":[float(np.quantile(B,.025)),float(np.quantile(B,.975))],
                "scope":"Sampling variability of sample-recording contrast; versions retained within recording, not acoustic truth"})
    dump(a.out/"aggregate_queries.json",answers);dump(a.out/"weight_identity.json",identity);dump(a.out/"bootstrap.json",boot)
    assert max(x["residual_max"] for x in identity)<1e-9,"Weight identity failed"
    scale=np.maximum(allmean[~hold].std(0),1e-12)
    rankings=[]
    for s,sm in split.items():
        universe=np.flatnonzero(sm)
        anchors=universe[np.argsort(h[universe],kind="stable")[:50]]
        for pn,vals in pol.items():
            support=sm&np.isfinite(vals).all(1);univ=np.flatnonzero(support)
            jacc=[];top1=[];spearman=[]
            for anchor in anchors:
                if not support[anchor]:continue
                eligible=univ[univ!=anchor]
                d0=np.sum(((allmean[eligible]-allmean[anchor])/scale)**2,axis=1)
                dp=np.sum(((vals[eligible]-vals[anchor])/scale)**2,axis=1)
                i0=eligible[np.argsort(d0,kind="stable")[:20]];ip=eligible[np.argsort(dp,kind="stable")[:20]]
                common=set(i0)&set(ip);union=set(i0)|set(ip)
                jacc.append(len(common)/len(union));top1.append(bool(i0[0]==ip[0]))
            rankings.append({"split":s,"policy":pn,"eligible_recordings":len(univ),"anchors_evaluated":len(jacc),
                             "mean_top20_jaccard":float(np.mean(jacc)) if jacc else None,"top1_agreement":float(np.mean(top1)) if top1 else None,
                             "reference":"Recording-allmean on the same support; each policy uses its own anchor representation"})
    dump(a.out/"ranking_queries.json",rankings)
    
    builds=bu[bo[:2]]
    m2=[];verify=[];coherence=[]
    for s,sm in split.items():
        mats=[];counts=[]
        for b in builds:
            bm=(build==b)&sm[ri];mm,cc=grouped(V[bm],ri[bm],R);mats.append(mm);counts.append(cc)
        common=sm&(counts[0]>0)&(counts[1]>0);pairvals=np.stack(mats)[:,common,:]
        for j,f in enumerate(FEATURES):
            vals=pairvals[:,:,j];rectlo=vals.min(0).sum();recthi=vals.max(0).sum();worlds=vals.sum(1)
            exactlo=worlds.min();exacthi=worlds.max()
            width=float(recthi-rectlo);cw=float(exacthi-exactlo)
            m2.append({"split":s,"feature":f,"common_recordings":int(common.sum()),"builds":[str(b) for b in builds],
              "rectangular_sum_bounds":[float(rectlo),float(recthi)],"shared_build_sum_bounds":[float(exactlo),float(exacthi)],
              "width_ratio":width/cw if cw>1e-12 else None,"width_reduction":1-cw/width if width>1e-12 else None,
              "semantics":"SUM of recording-within-build means on fixed common support. Global-build worlds are a stipulated query constraint, not inferred true audio.",
              "representation":"Independent record candidates allow mixed builds excluded by the global-build query"})
            if not common.any():continue
            
            widths=vals.max(0)-vals.min(0);N=len(widths);random=np.random.default_rng(SEED).permutation(N)
            for policy,order in {"width_first":np.argsort(-widths,kind="stable"),"random":random,"id_order":np.arange(N)}.items():
                for frac in [0,.01,.05,.1,.25,.5,1]:
                    k=int(math.ceil(frac*N));remaining=width-widths[order[:k]].sum()
                    verify.append({"split":s,"feature":f,"policy":policy,"budget":k,"candidates":N,"remaining_independent_model_width":max(0,float(remaining)),
                        "oracle":"Controlled revelation of observed build-0 candidate, not human verification"})
        
        keysA={};keysB={}
        for i in np.flatnonzero(sm[ri] & np.isin(build,builds) & (pay!="")):
            key=(str(rec[i]),str(pay[i]))
            (keysA if build[i]==builds[0] else keysB).setdefault(key,[]).append(i)
        shared=sorted(set(keysA)&set(keysB));uniq={}
        for key in shared:uniq.setdefault(key[0],key)
        selected=list(uniq.values())
        if selected:
            aa=np.array([V[keysA[k]].mean(0) for k in selected]);bb=np.array([V[keysB[k]].mean(0) for k in selected])
            for j,f in enumerate(FEATURES):
                dd=bb[:,j]-aa[:,j]
                coherence.append({"split":s,"feature":f,"matched_input_digest_recordings":len(selected),
                    "mean_build_contrast":float(dd.mean()),"median_absolute_contrast":float(np.median(np.abs(dd))),
                    "scope":"Same reported encoded MD5 per recording; no independent audio replay or cryptographic identity validation"})
    dump(a.out/"uncertainty_queries.json",m2);dump(a.out/"verification_queries.json",verify);dump(a.out/"matched_payload_build.json",coherence)
    
    exhaustive=[]
    rr=np.random.default_rng(SEED)
    for N in range(1,8):
        for trial in range(20):
            A=rr.integers(-10,11,(2,N)).astype(float)
            possible=np.array([sum(A[c,i] for i,c in enumerate(cs)) for cs in itertools.product(range(2),repeat=N)])
            rect=[A.min(0).sum(),A.max(0).sum()];world=A.sum(1);con=[world.min(),world.max()]
            assert rect==[possible.min(),possible.max()]
            assert rect[0]<=con[0]<=con[1]<=rect[1]
            exhaustive.append({"N":N,"trial":trial,"worlds":2**N,"rect_bounds":rect,"shared_build_bounds":con})
    dump(a.out/"exhaustive_correctness.json",{"cases":len(exhaustive),"checks":exhaustive,
      "counterexample":{"build_A":[0,100],"build_B":[100,0],"independent_sum_range":[0,200],"shared_build_sum_range":[100,100]},
      "novelty":"This is an established possible-world semantics example, not a new algorithm"})
    threshold_queries=[]
    control_counts=[]
    for s,sm in split.items():
        control_counts.append({"split":s,"policy":"all_submission","distinct_recordings":int(len(set(rec[sm[ri]]))),"raw_rows":int(sm[ri].sum())})
        for pn,vals in pol.items():
            sp=sm&np.isfinite(vals).all(1)
            control_counts.append({"split":s,"policy":pn,"distinct_recordings":int(sp.sum()),"fixed_support_expected":int(sp.sum())})
        rmin=np.full((R,len(FEATURES)),np.inf);rmax=np.full_like(rmin,-np.inf)
        np.minimum.at(rmin,ri,V);np.maximum.at(rmax,ri,V)
        for j,f in enumerate(FEATURES):
            for quantile in [.1,.5,.9]:
                threshold=float(np.quantile(allmean[~hold,j],quantile))
                lower=int((rmin[sm,j]>=threshold).sum());upper=int((rmax[sm,j]>=threshold).sum())
                refcount=int((allmean[sm,j]>=threshold).sum())
                for pn,vals in pol.items():
                    sp=sm&np.isfinite(vals).all(1)
                    lo=int((rmin[sp,j]>=threshold).sum());hi=int((rmax[sp,j]>=threshold).sum())
                    count=int((vals[sp,j]>=threshold).sum())
                    assert lo<=count<=hi
                    threshold_queries.append({"split":s,"feature":f,"development_quantile":quantile,"threshold":threshold,"policy":pn,
                        "eligible_recordings":int(sp.sum()),"count":count,"same_support_allmean_count":int((allmean[sp,j]>=threshold).sum()),
                        "observed_candidate_count_range":[lo,hi],"version_crossing_recordings":hi-lo,
                        "semantics":"One observed scalar value per recording; bounds over observed submissions on fixed support, with no audio truth"})
    assert all(r["distinct_recordings"]==int(split[r["split"]].sum()) for r in control_counts if r["policy"]!="recording_dominant_build")
    dump(a.out/"threshold_queries.json",threshold_queries);dump(a.out/"distinct_count_controls.json",control_counts)
    np.savez_compressed(a.out/"policy_recordings.npz",recording=ids,holdout=hold,year=years,n_sub=nsub,n_payload=npay,
                        feature=np.array(FEATURES),**pol)
    result={"data_source_sha256":{n:sha(a.data/n) for n in ["submission_features.npy","submission_metadata.npz","intake_receipt.json"]},
      "code_sha256":sha(Path(__file__)),"software":{"python":platform.python_version(),"numpy":np.__version__},
      "elapsed_seconds":time.monotonic()-start,"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      "outputs_sha256":{x.name:sha(x) for x in a.out.iterdir() if x.is_file() and x.name!="analysis_receipt.json"}}
    dump(a.out/"analysis_receipt.json",result)
    print(json.dumps({"scope":composition,"elapsed_seconds":result["elapsed_seconds"],"peak_rss_bytes":result["peak_rss_bytes"]}),flush=True)
if __name__=="__main__":main()
