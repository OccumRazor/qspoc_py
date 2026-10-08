import qutip
from scipy.sparse.linalg import eigsh
from . import propagation_method
from scipy.linalg import ishermitian

def KROTOV_CHEBY(H,state,dt,c_ops=None,backwards=False,initialize=False):
    if c_ops is None:
        c_ops = []
    if len(c_ops) > 0:
        raise NotImplementedError("Liouville exponentiation not implemented")
    assert isinstance(H, list) and len(H) > 0
    if isinstance(H[0], list):
        Ht = H[0][1] * H[0][0]
    else:
        Ht = H[0]
    for part in H[1:]:
        if isinstance(part, list):
            Ht += part[1] * part[0]
        else:
            Ht += part
    Ht = Ht.full()
    if not ishermitian(Ht):
        print('not Herm')
    eig_vals = eigsh(Ht,return_eigenvectors=False)
    E_max = max(eig_vals)
    E_min = min(eig_vals)
    return qutip.Qobj(propagation_method.Chebyshev(Ht,state.full(),E_max,E_min,dt,backwards = backwards))
