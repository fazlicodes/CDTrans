import torch
from tqdm import tqdm

def long_matmul(x,y, chunk_size=1000, topk=2, target_sample_num=2):

    knnidxs = torch.zeros(x.shape[0])
    target_knnsims = torch.zeros((topk, y.shape[1]))
    target_knnidxs = torch.zeros((topk, y.shape[1]))
    knnidx_topks = torch.zeros(x.shape[0],target_sample_num)
    target_knnsims, target_knnidxs = torch.zeros((topk, y.shape[1]), dtype=x.dtype, device=x.device), \
                             torch.zeros((topk, y.shape[1]), dtype=torch.long, device=x.device)
    for i in tqdm(range(0,x.shape[0],chunk_size)):

        chunk_size_i = min(chunk_size,x.shape[0]-i)

        chunk = torch.matmul(x[i:i+chunk_size_i],y)
        
        _, knnidx = torch.max(chunk,1)
        knnidxs[i:i+chunk_size_i] += knnidx


        target_knnsim, target_knnidx = torch.topk(chunk, k=topk, dim=0, largest=True, sorted=False)

        mask = target_knnsim > target_knnsims
        target_knnsims[mask] = target_knnsim[mask]
        target_knnidxs[mask] = target_knnidx[mask]+i

        _, knnidx_topk = torch.topk(chunk,k=target_sample_num,dim=1)
        knnidx_topks[i:i+chunk_size_i,:] += knnidx_topk
    if topk==1:
        target_knnsims = target_knnsims.flatten()
        target_knnidxs = target_knnidxs.flatten()

    return knnidxs, target_knnsims, target_knnidxs, knnidx_topks

x = torch.randn(2000,384)
y = torch.randn(384, 4000)

c = torch.matmul(x,y)

target_knnsim, target_knnidx = torch.topk(c, k=1, dim=0, largest=True, sorted=False)
knnidxs, target_knnsims1, target_knnidxs1, knnidx_topks = long_matmul(x,y, chunk_size=1000, topk=1, target_sample_num=2)


print((target_knnidxs1 == target_knnidx).sum())