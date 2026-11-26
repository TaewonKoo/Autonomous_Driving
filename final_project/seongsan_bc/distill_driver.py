"""Train a fully learned nine-action policy with two supervised decision heads."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from learned_driver import OUT, structured_action


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.longitudinal=nn.Sequential(nn.Linear(27,96),nn.Tanh(),nn.Linear(96,64),nn.Tanh(),nn.Linear(64,3))
        self.lateral=nn.Sequential(nn.Linear(27,48),nn.Tanh(),nn.Linear(48,32),nn.Tanh(),nn.Linear(32,3))

    def forward(self, x):
        v=x[:,0]
        controls=torch.amin(x[:,[9,10]],dim=1)
        speed_squared=v.square()
        columns=[v/10,x[:,1]/14,x[:,2],x[:,3]/20,x[:,4]/20,x[:,5]/10,x[:,6]/10,
                 x[:,7]/20,x[:,8]/10,controls/20,x[:,11],x[:,12]/30,x[:,15]/12,
                 x[:,16]/20,(x[:,7]-(speed_squared-x[:,8].square())/5)/10,
                 (controls-speed_squared/5)/8,(x[:,16]-speed_squared/5)/8,
                 (x[:,3]-torch.relu(-x[:,5]))/10,(x[:,4]-torch.relu(x[:,6]))/10,
                 (x[:,7]-v*.45-(speed_squared-x[:,8].square())/5)/5,
                 (v<.01).float(),(x[:,7]<2.5).float(),(controls<5.).float(),
                 (x[:,16]<5.).float(),(x[:,15]<6.5).float(),
                 (x[:,3]<99.).float(),(x[:,4]<99.).float()]
        z=torch.stack(columns,dim=1).clamp(-3.,3.)
        lon=self.longitudinal(z)
        lat=self.lateral(z)
        return (lon[:,:,None]+lat[:,None,:]).flatten(1)


def fit(name, tags, seed=25,epochs=35, augment=False):
    torch.manual_seed(seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    samples=[]
    for tag in tags:
        for path in sorted((OUT/tag).glob("*/labels.npz")):
            with np.load(path) as data:
                samples.append((data["x"],data["y"]))
    if not samples:
        raise ValueError("No completed training batches")
    x_array=np.concatenate([x for x,y in samples])
    y_array=np.concatenate([y for x,y in samples])
    if augment:
        rng=np.random.default_rng(seed)
        # Offline examples near side-gap and braking boundaries, without editing SUMO.
        nearby=x_array[rng.integers(len(x_array),size=40000)].copy()
        nearby[:,0]=rng.uniform(0,12,len(nearby))
        nearby[:,2]=rng.choice([-1,0,1],size=len(nearby))
        for col in (3,4):
            nearby[:,col]=rng.uniform(0,25,len(nearby))
        nearby[:,5:7]=rng.uniform(-8,8,(len(nearby),2))
        nearby[:,15]=rng.uniform(0,30,len(nearby))
        p=[7,2.6,7,9,1.2,.8,3.]
        y_extra=np.asarray([structured_action(f,p) for f in nearby],dtype=np.int64)
        x_array=np.concatenate((x_array,nearby))
        y_array=np.concatenate((y_array,y_extra))
    xs=torch.tensor(x_array,device=device)
    ys=torch.tensor(y_array,device=device)
    lon,lat=ys//3,ys%3
    model=Student().to(device)
    optim=torch.optim.Adam(model.parameters(),lr=.001)
    weights=[t.bincount(minlength=3).clamp(min=1).float().pow(-.35) for t in (lon,lat)]
    for ep in range(epochs):
        for ids in torch.randperm(len(xs),device=device).split(512):
            pred=model(xs[ids]).reshape(-1,3,3)
            loss=nn.functional.cross_entropy(pred[:,:,0],lon[ids],weight=weights[0])
            loss+=nn.functional.cross_entropy(pred[:,0,:],lat[ids],weight=weights[1])
            optim.zero_grad()
            loss.backward()
            optim.step()
        if ep in (int(.7*epochs),int(.9*epochs)):
            optim.param_groups[0]["lr"]*=.3
    torch.jit.script(model.cpu().eval()).save(str(OUT/(name+".pt")))
    print(name,len(xs), flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("tags",nargs="+")
    parser.add_argument("--epochs",type=int, default=35)
    parser.add_argument("--seed",type=int,default=25)
    parser.add_argument("--augment",action="store_true")
    args=parser.parse_args()
    fit(args.name,args.tags,args.seed,args.epochs,args.augment)
