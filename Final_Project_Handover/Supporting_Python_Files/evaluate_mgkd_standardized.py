#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
from ultralytics import RTDETR

BASE=Path('/Users/kalamanjunatha/Downloads/sugarbeet')
MODEL=BASE/'Experiments/Hybrid_Approaches/MGKD/results/MGKD_YOLOv11_RTDETR_70_20_10/last.pt'
IMAGES=BASE/'dataset_70_20_10_detection/images/test'
LABELS=BASE/'dataset_70_20_10_detection/labels/test'
OUT=BASE/'Experiments/Unified_Standardized_Evaluation/MGKD_YOLOv11_RTDETR_70_20_10'
RAW_CONF=0.001
CONF=0.25
IOU=0.50
PREDICT_IOU=0.70
IMG=640

def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    ab=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    u=aa+ab-inter
    return inter/u if u>0 else 0.

def evaluate_prf(preds,gts,thr=.5):
    tp=fp=fn=0
    for ps,gs in zip(preds,gts):
        ps=sorted(ps,key=lambda x:x[4],reverse=True)
        used=set()
        for p in ps:
            best=-1; bi=0.
            for j,g in enumerate(gs):
                if j in used: continue
                v=iou(p,g)
                if v>bi: bi=v; best=j
            if best>=0 and bi>=thr:
                tp+=1; used.add(best)
            else: fp+=1
        fn += len(gs)-len(used)
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f1=2*p*r/(p+r) if p+r else 0.
    return p,r,f1,tp,fp,fn

def ap(preds,gts,thr):
    total_gt=sum(len(x) for x in gts)
    if total_gt==0: return 0.
    flat=[]
    for i,(ps,gs) in enumerate(zip(preds,gts)):
        for p in ps: flat.append((p[4],i,p))
    flat.sort(key=lambda x:x[0],reverse=True)
    used=[set() for _ in gts]
    tp=np.zeros(len(flat)); fp=np.zeros(len(flat))
    for k,(_,i,p) in enumerate(flat):
        best=-1; bi=0.
        for j,g in enumerate(gts[i]):
            if j in used[i]: continue
            v=iou(p,g)
            if v>bi: bi=v; best=j
        if best>=0 and bi>=thr:
            tp[k]=1.; used[i].add(best)
        else: fp[k]=1.
    ctp=np.cumsum(tp); cfp=np.cumsum(fp)
    rec=ctp/total_gt
    prec=ctp/np.maximum(ctp+cfp,1e-12)
    for k in range(len(prec)-2,-1,-1): prec[k]=max(prec[k],prec[k+1])
    rp=np.concatenate(([0.],rec,[1.])); pp=np.concatenate(([1.],prec,[0.]))
    return float(np.sum((rp[1:]-rp[:-1])*pp[1:]))

def get_gts(paths):
    out=[]
    for p in paths:
        g=[]; lp=LABELS/f'{p.stem}.txt'
        if lp.exists():
            for line in lp.read_text().splitlines():
                v=line.split()
                if len(v)<5: continue
                try: cls=int(float(v[0])); cx,cy,w,h=map(float,v[1:5])
                except: continue
                if cls!=0: continue
                g.append([max(0.,cx-w/2),max(0.,cy-h/2),min(1.,cx+w/2),min(1.,cy+h/2),1.,0])
        out.append(g)
    return out

def get_preds(model,paths):
    out=[]
    for i,p in enumerate(paths,1):
        r=model.predict(source=str(p),imgsz=IMG,conf=CONF,verbose=False)[0]
        cur=[]
        if r.boxes is not None and len(r.boxes):
            boxes=r.boxes.xyxyn.cpu().numpy(); scores=r.boxes.conf.cpu().numpy(); cls=r.boxes.cls.cpu().numpy()
            for b,s,c in zip(boxes,scores,cls):
                if int(c)!=0: continue
                cur.append([float(np.clip(b[0],0,1)),float(np.clip(b[1],0,1)),float(np.clip(b[2],0,1)),float(np.clip(b[3],0,1)),float(s),0])
        out.append(cur)
        if i==1 or i%10==0 or i==len(paths): print(f'Processing {i}/{len(paths)}')
    return out

def main():
    paths=sorted([p for p in IMAGES.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}])
    if not MODEL.exists(): raise FileNotFoundError(MODEL)
    print('='*80); print('MGKD — UNIFIED STANDARDIZED EVALUATION'); print('='*80)
    print('Checkpoint:',MODEL); print('Test images:',len(paths)); print('Conf:',CONF,'IoU:',IOU,'imgsz:',IMG)
    model=RTDETR(str(MODEL))
    gts=get_gts(paths)
    preds=get_preds(model,paths)
    p,r,f1,tp,fp,fn=evaluate_prf(preds,gts,IOU)
    aps=[ap(preds,gts,float(t)) for t in np.arange(.5,1.,.05)]
    m50=aps[0]; m5095=float(np.mean(aps))
    result={'model':'MGKD_YOLOv11_RTDETR','split':'70_20_10','test_images':len(paths),'confidence':CONF,'iou':IOU,'precision':p,'recall':r,'f1':f1,'mAP50':m50,'mAP50_95':m5095,'TP':tp,'FP':fp,'FN':fn,'gt_objects':sum(map(len,gts)),'predictions':sum(map(len,preds)),'checkpoint':str(MODEL)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'MGKD_standardized_results.json').write_text(json.dumps(result,indent=2))
    (OUT/'MGKD_standardized_results.txt').write_text('\n'.join([f'Model: MGKD YOLOv11 -> RT-DETR', 'Split: 70_20_10', f'Test images: {len(paths)}', f'Confidence: {CONF}', f'IoU: {IOU}', f'Precision: {p:.6f}', f'Recall: {r:.6f}', f'F1: {f1:.6f}', f'mAP50: {m50:.6f}', f'mAP50-95: {m5095:.6f}', f'TP: {tp}', f'FP: {fp}', f'FN: {fn}']))
    print('\nFINAL MGKD STANDARDIZED RESULTS')
    print(f'Precision : {p:.4f}'); print(f'Recall    : {r:.4f}'); print(f'F1        : {f1:.4f}'); print(f'mAP50     : {m50:.4f}'); print(f'mAP50-95  : {m5095:.4f}'); print(f'TP/FP/FN  : {tp} / {fp} / {fn}')
    print('\nSaved:',OUT)
if __name__=='__main__': main()
