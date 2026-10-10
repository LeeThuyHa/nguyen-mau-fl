"""NumPy GRU/FedAvg engineering prototype. Synthetic feature sequences, NOT pig video.
Run: python run_experiment.py --out results
Only NumPy is required. No image detector, MobileNet, network service or privacy guarantee.
"""
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np

T, D, H, K = 12, 8, 12, 3
ROUNDS, BATCH, LR = 20, 18, .15
NAMES = ('stable', 'linear_drift', 'oscillation')

def digest(a): return hashlib.sha256(a.tobytes()).hexdigest()
def init(seed):
    g=np.random.default_rng(seed); p={}
    for k in ('z','r','n'):
        p['W'+k]=g.normal(0,.22,(D,H));p['U'+k]=g.normal(0,.22,(H,H));p['b'+k]=np.zeros(H)
    p['Wo']=g.normal(0,.22,(H,K));p['bo']=np.zeros(K)
    return p
def copy(p): return {k:v.copy() for k,v in p.items()}
def sigmoid(a): return 1/(1+np.exp(-np.clip(a,-40,40)))

def forward(p,x):
    h=np.zeros((len(x),H)); cache=[]
    for xt in x.transpose(1,0,2):
        old=h; z=sigmoid(xt@p['Wz']+old@p['Uz']+p['bz'])
        r=sigmoid(xt@p['Wr']+old@p['Ur']+p['br'])
        n=np.tanh(xt@p['Wn']+(r*old)@p['Un']+p['bn'])
        h=(1-z)*n+z*old;cache.append((xt,old,z,r,n))
    logits=h@p['Wo']+p['bo']; logits-=logits.max(axis=1,keepdims=True)
    prob=np.exp(logits);prob/=prob.sum(axis=1,keepdims=True)
    return prob,h,cache

def loss_grad(p,x,y):
    prob,h,cache=forward(p,x)
    loss=-np.log(np.maximum(prob[np.arange(len(y)),y],1e-12)).mean()
    dl=prob.copy();dl[np.arange(len(y)),y]-=1;dl/=len(y)
    g={k:np.zeros_like(v) for k,v in p.items()}
    g['Wo']=h.T@dl;g['bo']=dl.sum(0);dh=dl@p['Wo'].T
    for xt,old,z,r,n in reversed(cache):
        an=dh*(1-z)*(1-n*n);az=dh*(old-n)*z*(1-z)
        drh=an@p['Un'].T;ar=drh*old*r*(1-r)
        for k,a,u in [('n',an,r*old),('z',az,old),('r',ar,old)]:
            g['W'+k]+=xt.T@a;g['U'+k]+=u.T@a;g['b'+k]+=a.sum(0)
        dh=dh*z+drh*r+az@p['Uz'].T+ar@p['Ur'].T
    return float(loss),g

def epoch(p,x,y,rng):
    order=rng.permutation(len(y))
    for start in range(0,len(y),BATCH):
        idx=order[start:start+BATCH];_,g=loss_grad(p,x[idx],y[idx])
        norm=np.sqrt(sum(np.sum(a*a) for a in g.values()));scale=min(1.,5/max(norm,1e-12))
        for k in p:p[k]-=LR*scale*g[k]

def aggregate(models,counts):
    if not models or len(models)!=len(counts) or any(n<=0 for n in counts):raise ValueError('Invalid clients/counts')
    keys=set(models[0])
    if any(set(m)!=keys or any(m[k].shape!=models[0][k].shape for k in keys) for m in models):raise ValueError('Incompatible parameters')
    if any(not np.isfinite(v).all() for m in models for v in m.values()):raise ValueError('Non-finite update')
    return {k:sum(n*m[k] for n,m in zip(counts,models))/sum(counts) for k in models[0]}

def data(seed,counts,projection):
    rng=np.random.default_rng(seed);xs=[];ys=[]; t=np.linspace(-1,1,T)
    for c,num in enumerate(counts):
        for _ in range(num):
            base=rng.normal(0,.16,4);raw=np.tile(base,(T,1))
            amp=rng.uniform(.7,1.2)
            if c==1:raw[:,0]+=amp*t;raw[:,1]+=.5*amp*t
            elif c==2:raw[:,0]+=amp*np.sin(2*np.pi*(t+1));raw[:,1]+=amp*np.cos(2*np.pi*(t+1))
            raw+=rng.normal(0,.07,(T,4))
            xs.append(np.tanh(raw@projection));ys.append(c)
    idx=rng.permutation(len(ys));return np.array(xs)[idx],np.array(ys)[idx]

def evaluate(p,x,y):
    probs,_,_=forward(p,x);pred=probs.argmax(1);cm=np.zeros((K,K),int)
    np.add.at(cm,(y,pred),1);f=[]
    for c in range(K):
        tp=cm[c,c];den=cm[:,c].sum()+cm[c,:].sum();f.append(2*tp/den if den else 0.)
    return {'accuracy':float((pred==y).mean()),'macro_f1':float(np.mean(f)),
            'loss':float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean()),'confusion':cm.tolist()}

def checks():
    p=init(9);rng=np.random.default_rng(8);x=rng.normal(0,.2,(2,T,D));y=np.array([0,2])
    _,g=loss_grad(p,x,y);errors=[]
    for k in p:
        index=tuple(0 for _ in p[k].shape);original=p[k][index];eps=1e-5
        p[k][index]=original+eps;a=loss_grad(p,x,y)[0]
        p[k][index]=original-eps;b=loss_grad(p,x,y)[0];p[k][index]=original
        errors.append(abs((a-b)/(2*eps)-g[k][index]))
    assert max(errors)<1e-6,errors
    a={'v':np.array([1.,3.])};b={'v':np.array([5.,7.])}
    assert np.allclose(aggregate([a,b],[1,3])['v'],[4.,6.])
    rejected=False
    try:aggregate([a,{'v':np.zeros(3)}],[1,1])
    except ValueError:rejected=True
    assert rejected
    x=np.tile(x[:1],(12,1,1));y=np.zeros(12,dtype=int);before=loss_grad(p,x,y)[0]
    for _ in range(20):epoch(p,x,y,rng)
    assert loss_grad(p,x,y)[0]<before
    return {'gradient_check_max_abs_error':max(errors),'gradient_coordinates_checked':len(errors),
            'weighted_aggregation':'PASS','reject_shape_mismatch':'PASS','small_batch_loss_decreases':'PASS'}

def run(out):
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();test_results=checks()
    rows=[];curves=[];counts=[[60,20,10],[10,60,20],[20,10,60]]
    projection=np.random.default_rng(123).normal(0,.7,(4,D));fingerprint=digest(projection)
    for seed in [11,22,33]:
        clients=[data(seed*100+i,c,projection) for i,c in enumerate(counts)]
        tx,ty=data(90000+seed,[60,60,60],projection)
        trainx=np.concatenate([v[0] for v in clients]);trainy=np.concatenate([v[1] for v in clients])
        train_hashes={digest(a) for a in trainx};assert not any(digest(a) in train_hashes for a in tx)
        initial=init(seed)
        for method in ['local_0','local_1','local_2','centralized','fedavg']:
            p=copy(initial);rng=np.random.default_rng(seed);tick=time.perf_counter();upload=download=0
            for rnd in range(1,ROUNDS+1):
                if method=='fedavg':
                    updated=[]
                    for x,y in clients:
                        cp=copy(p);epoch(cp,x,y,rng);updated.append(cp)
                    p=aggregate(updated,[len(y) for _,y in clients])
                    size=sum(a.nbytes for a in p.values());upload+=size*len(clients);download+=size*len(clients)
                else:
                    x,y=(trainx,trainy) if method=='centralized' else clients[int(method[-1])]
                    epoch(p,x,y,rng)
                loss,_=loss_grad(p,trainx,trainy)
                curves.append({'seed':seed,'method':method,'round':rnd,'global_train_loss':loss})
            elapsed=time.perf_counter()-tick;result=evaluate(p,tx,ty)
            result.update(seed=seed,method=method,seconds=elapsed,upload_bytes=upload,download_bytes=download)
            rows.append(result)
            print(f'{seed} {method}: macro_f1={result["macro_f1"]:.4f}, {elapsed:.2f}s',flush=True)
            if method=='fedavg':np.savez(out/f'fedavg_seed{seed}.npz',**p,projection=projection)
        assert digest(projection)==fingerprint
    test_results['frozen_projection_unchanged']='PASS';test_results['no_identical_train_test_sequences']='PASS'
    summaries={}
    for m in ['local_0','local_1','local_2','centralized','fedavg']:
        rs=[r for r in rows if r['method']==m]
        summaries[m]={k:{'mean':float(np.mean([r[k] for r in rs])), 'std':float(np.std([r[k] for r in rs],ddof=1))} for k in ['macro_f1','accuracy','seconds']}
    params=sum(v.size for v in initial.values())
    info={'warning':'SYNTHETIC FEATURE SEQUENCES ONLY. No video, MobileNet, detector, real farm or network experiment.',
          'environment':{'python':sys.version,'numpy':np.__version__,'os':platform.platform(),'processor':platform.processor()},
          'config':{'seeds':[11,22,33],'rounds':ROUNDS,'batch':BATCH,'learning_rate':LR,'optimizer':'SGD gradient norm clip 5',
          'sequence_length':T,'input_dim':D,'hidden_dim':H,'classes':NAMES,'client_class_counts':counts,'test_class_counts':[60]*3,
          'trainable_parameters':params,'projection_parameters':32,'dtype':'float64','feature_projection_sha256':fingerprint},
          'checks':test_results,'results':rows,'summary':summaries,'total_seconds':time.perf_counter()-start}
    (out/'results.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'learning_curves.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]));writer.writeheader();writer.writerows(curves)
    print('Saved',out.resolve(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    run(parser.parse_args().out)
