import numpy as np

ID=[[1,0],[0,1]]
ket=[[[1],[0]],[[0],[1]],ID]

def D2B(n,N):
    # n current number
    # N maximum
    # to make sure the output have the same length
    l=int(np.log2(N))
    opt=[]
    for i in range(l):
        if n <2**(l-i-1):opt.append(0)
        else:
            opt.append(1)
            n-=2**(l-i-1)
    return opt

def densityMatrix(state):
    if isinstance(state[0], list) or isinstance(state[0], np.ndarray):
        state = list(np.array(state).T[0])
    dm = np.zeros([len(state),len(state)],dtype=np.complex128)
    for i in range(len(state)):
        for j in range(len(state)):
            dm[i][j] = complex(state[i] * np.conjugate(state[j]))
    return np.array(dm)

def partialTrace(state,idle):
    # idle here refers to those traced qubits.
    n=int(np.log2(len(state)))
    traceList=[i for i in range(n) if i not in idle]
    li=[]
    for i in range(n):
        if i in traceList:li.append(0)
        else:li.append(1)
    N=2**sum(li) # number of basis
    rho=densityMatrix(state)
    res=np.complex128(np.zeros([2**len(traceList),2**len(traceList)]))
    for i in range(N):
        li=D2B(i,N)
        for j in traceList:
            li.insert(j,2)
        basis=ket[li[-1]]
        for k in range(n-2,-1,-1):
            basis=np.kron(ket[li[k]],basis)
        res+=np.matmul(np.transpose(basis),np.matmul(rho,basis))   
    return res

def completes(ele,otherPart,N):
    # this function would be called by genPartition(), 
    # to check whether a term should be added to idle
    for i in range(len(otherPart)):
        if len(ele)+len(otherPart[i])==N:
            temp=ele+otherPart[i]
            count=0
            for j in range(N):
                if j in temp:count+=1
            if count>=N:return False
        if len(ele)==len(otherPart[i]):
            count1=0
            for j in range(len(ele)):
                if ele[j] in otherPart[i]:count1+=1
            if count1==len(ele):return False
    return True

def genPartition(N):
    # for n qubits, 2^(n-1)-1 terms for n>2
    n=int(N/2)
    li=[[i] for i in range(N)]
    idles=[[i] for i in range(N)]
    for i in range(1,n+1):
        temp=[[j] for j in range(N)]
        for j in range(i-1):
            for k in range(len(temp)):
                if len(temp[k])==j+1:
                    for l in range(N):
                        if li[l][0] not in temp[k]:temp.append(temp[k]+li[l])
        for j in range(N,len(temp)):
            if completes(temp[j],idles,N) and temp[j] not in idles: idles.append(temp[j])
    return idles

