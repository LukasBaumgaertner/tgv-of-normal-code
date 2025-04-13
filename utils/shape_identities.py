from dolfin import as_tensor, dot, grad, div, inner, outer, sqrt, as_vector, le, conditional, acos, tr, CellVolume, FacetArea, Constant, CellNormal
from ufl import Identity as identity, atan_2, indices


def sign(a):
    return conditional(le(a , -a), -1, 1)

def vnorm(vec):
    return sqrt(inner(vec, vec))

def stableangle(a,b):
    mu = conditional( vnorm(a-b) >= vnorm(b), vnorm(a-b) - (vnorm(a)-vnorm(b)), vnorm(b) - ( vnorm(a)-vnorm(a-b)) )
    return 2*atan_2( sqrt( ((vnorm(a)-vnorm(b)) + vnorm(a-b) )*mu) , sqrt( (vnorm(a)+(vnorm(b) + vnorm(a-b)) )*((vnorm(a)- vnorm(a-b)) +vnorm(b)  )) )

def angle(a,b):
    return acos(conditional(le(inner(a,b), 1), inner(a,b), 1))

def angle2(a,b):
    return 2*atan_2(sqrt(inner(a-b, a-b)), sqrt(inner(a+b, a+b)))

def logmu(n, mu):
    return sign(inner(n('-'), mu('+')))*angle2(n('+'), n('-'))

def dlogmu(n, mu, V):
    return -inner(dn(n, V)('+'), mu('+')) - inner(dn(n, V)('-'), mu('-'))

def ddlogmu(n, mu, tE, V, W):
    return -inner(dn(n, V)('+'), dmu(mu, tE, W)('+')) - inner(dn(n, V)('-'), dmu(mu, tE, W)('-')) - inner(ddn(n, V, W)('+'), mu('+')) - inner(ddn(n, V, W)('-'), mu('-'))

def log(n, mu, side="+"):
    return logmu(n, mu)*mu(side)

def dlog(n, mu, tE, V, side = "+"):
    return dlogmu(n, mu, V)*mu(side) + logmu(n, mu)*dmu(mu, tE, V)(side)

def ddlog(n, mu, tE, V, W, side = "+"):
    return ddlogmu(n, mu, tE, V, W)*mu(side) + dlogmu(n, mu, W)*dmu(mu, tE, V)(side) + dlogmu(n, mu, V)*dmu(mu, tE, W)(side) + logmu(n, mu)*ddmu(mu, tE, V, W)(side)

def tan_jump(u, tE, mu):
    n = CellNormal(u.function_space().mesh())
    return dot(u('+'), tE('+')) - parallelTransport(dot(u('-'), tE('+')), n('-'), n('+'))
    return dot(u('+'), tE('+')) + mu('+')*inner(mu('-'),dot(u('-'), tE('+'))) - tE('+')*inner(tE('+'),dot(u('-'), tE('+')))


def dtan_jump(u, tE, mu, V):
    ret = dot(dRT(u, V)('+'), tE('+')) + dot(u('+'), dtE(tE, V)('+'))
    ret += dmu(mu, tE, V)('+')*inner(mu('-'),dot(u('-'), tE('+'))) + mu('+')*inner(dmu(mu, tE, V)('-'),dot(u('-'), tE('+')))
    ret += mu('+')*inner(mu('-'),dot(dRT(u, V)('-'), tE('+'))) + mu('+')*inner(mu('-'),dot(u('-'), dtE(tE, V)('+')))
    ret += - dtE(tE, V)('+')*inner(tE('+'),dot(u('-'), tE('+'))) - tE('+')*inner(dtE(tE,V)('+'),dot(u('-'), tE('+')))
    ret += -tE('+')*inner(tE('+'),dot(dRT(u, V)('-'), tE('+'))) - tE('+')*inner(tE('+'),dot(u('-'), dtE(tE, V)('+')))
    return ret

def ddtan_jump(u, tE, mu, n, V, W):
    ret = dot(ddRT(u, n, V, W)('+'), tE('+')) + dot(dRT(u, V)('+'), dtE(tE, W)('+')) + dot(dRT(u, W)('+'), dtE(tE, V)('+')) + dot(u('+'), ddtE(tE, V, W)('+'))
    ret += ddmu(mu, tE, V, W)('+')*inner(mu('-'),dot(u('-'), tE('+'))) + mu('+')*inner(ddmu(mu, tE, V, W)('-'),dot(u('-'), tE('+')))
    ret += mu('+')*inner(mu('-'),dot(ddRT(u, n, V, W)('-'), tE('+'))) + mu('+')*inner(mu('-'),dot(u('-'), ddtE(tE, V, W)('+')))
    ret += dmu(mu, tE, W)('+')*(inner(dmu(mu, tE, V)('-'), dot(u('-'), tE('+'))) + inner(mu('-'), dot(dRT(u,V)('-'), tE('+'))) + inner(mu('-'), dot(u('-'), dtE(tE, V)('+'))))
    ret += dmu(mu, tE, V)('+')*(inner(dmu(mu, tE, W)('-'), dot(u('-'), tE('+'))) + inner(mu('-'), dot(dRT(u,W)('-'), tE('+'))) + inner(mu('-'), dot(u('-'), dtE(tE, W)('+'))))
    ret += mu('+')*(inner(dmu(mu, tE, V)('-'),dot(dRT(u,W)('-'), tE('+'))) + inner(dmu(mu, tE, W)('-'),dot(dRT(u,V)('-'), tE('+'))))
    ret += mu('+')*(inner(dmu(mu, tE, V)('-'),dot(u('-'), dtE(tE, W)('+'))) + inner(dmu(mu, tE, W)('-'),dot(u('-'), dtE(tE, V)('+'))))
    ret += mu('+')*(inner(mu('-'),dot(dRT(u,V)('-'), dtE(tE, W)('+'))) + inner(mu('-'),dot(dRT(u,W)('-'), dtE(tE, V)('+'))))

    ret += - ddtE(tE, V, W)('+')*inner(tE('+'),dot(u('-'), tE('+'))) - tE('+')*inner(ddtE(tE, V, W)('+'),dot(u('-'), tE('+')))
    ret += - tE('+')*inner(tE('+'),dot(ddRT(u, n, V, W)('-'), tE('+'))) - tE('+')*inner(tE('+'),dot(u('-'), ddtE(tE, V, W)('+')))
    ret += - dtE(tE,V)('+')*(inner(dtE(tE,W)('+'),dot(u('-'), tE('+'))) + inner(tE('+'),dot(dRT(u,W)('-'), tE('+'))) + inner(tE('+'),dot(u('-'), dtE(tE,W)('+'))))
    ret += - dtE(tE,W)('+')*(inner(dtE(tE,V)('+'),dot(u('-'), tE('+'))) + inner(tE('+'),dot(dRT(u,V)('-'), tE('+'))) + inner(tE('+'),dot(u('-'), dtE(tE,V)('+'))))
    ret += -tE('+')*(inner(dtE(tE,W)('+'),dot(dRT(u,V)('-'), tE('+'))) + inner(dtE(tE,V)('+'),dot(dRT(u,W)('-'), tE('+'))))
    ret += -tE('+')*(inner(dtE(tE,W)('+'),dot(u('-'), dtE(tE,V)('+'))) + inner(dtE(tE,V)('+'),dot(u('-'), dtE(tE,W)('+'))))
    ret += -tE('+')*(inner(tE('+'),dot(dRT(u,W)('-'), dtE(tE,V)('+'))) + inner(tE('+'),dot(dRT(u,V)('-'), dtE(tE,W)('+'))))
    return ret

# Parallel transport
def parallelTransport(xi, n_0, n):
    #return xi
    return xi - inner(xi, n)*(n_0+n)/(1+inner(n_0, n))

def dparallelTransport(xi,n_0,n, V):
    #return Constant((0.0,0.0,0.0))*V[0]
    return -inner(xi, dn(n, V))*(n+n_0)/(1+inner(n_0,n)) - inner(xi, n)*(dn(n,V)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, dn(n,V))/((1+inner(n_0,n))**2))

def ddparallelTransport(xi, n_0, n, V, W):
    #return Constant((0.0,0.0,0.0))*inner(V,W)
    ret = -inner(xi, ddn(n, V, W))*(n+n_0)/(1+inner(n_0,n))
    ret += -inner(xi, n)*(ddn(n,V,W)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, ddn(n,V,W))/((1+inner(n_0,n))**2))
    ret += -inner(xi, dn(n, W))*(dn(n,V)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, dn(n,V))/((1+inner(n_0,n))**2))
    ret += -inner(xi, dn(n, V))*(dn(n,W)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, dn(n, W)) / ((1 + inner(n_0, n)) ** 2))
    ret += inner(xi, n)*( (dn(n, V)*inner(n_0, dn(n, W))+ dn(n, W)*inner(n_0, dn(n, V)))/((1+inner(n_0, n))**2) )
    ret += -2*inner(xi, n)*(n_0+n)*inner(n_0, dn(n,V))*inner(n_0, dn(n,W))/((1+inner(n_0, n))**3)
    return ret


def parallelTransportMat(n_0, n):
    #return identity(3)
    return identity(3) - outer(n_0+n, n)/(1+inner(n_0, n))

def dparallelTransportMat(n_0,n, V):
    #return 0*identity(3)*V[0]
    return -outer(n+n_0, dn(n, V))/(1+inner(n_0,n)) - outer(dn(n,V), n)/(1+inner(n_0,n)) + outer((n+n_0)*inner(n_0, dn(n,V)), n)/((1+inner(n_0,n))**2)


def ddparallelTransportMat(n_0, n, V, W):
    #return 0*identity(3)*inner(V,W)
    ret = -outer(n+n_0, ddn(n, V, W))/(1+inner(n_0,n))
    ret += -outer(ddn(n,V,W)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, ddn(n,V,W))/((1+inner(n_0,n))**2), n)
    ret += -outer(dn(n,V)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, dn(n,V))/((1+inner(n_0,n))**2), dn(n, W))
    ret += -outer(dn(n,W)/(1+inner(n_0, n)) - (n+n_0)*inner(n_0, dn(n, W)) / ((1 + inner(n_0, n)) ** 2), dn(n, V))
    ret += outer( (dn(n, V)*inner(n_0, dn(n, W))+ dn(n, W)*inner(n_0, dn(n, V)))/((1+inner(n_0, n))**2), n )
    ret += -2*outer(n_0+n, n)*inner(n_0, dn(n,V))*inner(n_0, dn(n,W))/((1+inner(n_0, n))**3)
    return ret

def parallelTransportTensor(T, n_0, n):
    #return T
    P = parallelTransportMat(n_0, n)
    a,b,c,d,e,f = indices(6)
    return as_tensor(T[a,b,c]*P[d,a]*P[e,b]*P[f,c], (d,e,f))

def dparallelTransportTensor(T, n_0, n, V):
    #return 0*T*V[0]
    P = parallelTransportMat(n_0, n)
    dP = dparallelTransportMat(n_0, n, V)
    a,b,c,d,e,f = indices(6)
    return as_tensor(T[a,b,c]*dP[d,a]*P[e,b]*P[f,c], (d,e,f)) + as_tensor(T[a,b,c]*P[d,a]*dP[e,b]*P[f,c], (d,e,f)) + as_tensor(T[a,b,c]*P[d,a]*P[e,b]*dP[f,c], (d,e,f))

def ddparallelTransportTensor(T, n_0, n, V, W):
    #return 0*T*inner(V,W)
    P = parallelTransportMat(n_0, n)
    dPV = dparallelTransportMat(n_0, n, V)
    dPW = dparallelTransportMat(n_0, n, W)
    ddP = ddparallelTransportMat(n_0, n, V, W)
    a,b,c,d,e,f = indices(6)
    ret = as_tensor(T[a,b,c]*ddP[d,a]*P[e,b]*P[f,c] + T[a,b,c]*P[d,a]*ddP[e,b]*P[f,c] + T[a,b,c]*P[d,a]*P[e,b]*ddP[f,c], (d,e,f))
    ret += as_tensor(T[a,b,c]*dPV[d,a]*dPW[e,b]*P[f,c] + T[a,b,c]*dPW[d,a]*dPV[e,b]*P[f,c] + T[a,b,c]*dPW[d,a]*P[e,b]*dPV[f,c], (d,e,f))
    ret += as_tensor(T[a,b,c]*dPV[d,a]*P[e,b]*dPW[f,c] + T[a,b,c]*P[d,a]*dPV[e,b]*dPW[f,c] + T[a,b,c]*P[d,a]*dPW[e,b]*dPV[f,c], (d,e,f))
    return ret


#mesh quantities
def grad_tan(n, V):
    return grad(V) - outer(dot(grad(V), n),n)

def dn(n,V, surface=True):
    if surface:
        return -dot(grad(V).T,n)
    else:
        return -dot(grad_tan(n, V).T,n)
    
def ddn(n, V, W, surface = True):
    if surface:
        return dot(grad(V).T, dot(grad(W).T, n)) + dot(grad(W).T, dot(grad(V).T, n)) - n*inner( dot(grad(V).T, n), dot(grad(W).T, n))
    else:
        return dot(grad_tan(n,V).T, dot(grad_tan(n,W).T, n)) + dot(grad_tan(n,W).T, dot(grad_tan(n,V).T, n)) - n*inner( dot(grad_tan(n,V).T, n), dot(grad_tan(n,W).T, n))
        
def dtE(tE, V):
    return dot(grad(V), tE) - tE*inner(tE, dot(grad(V),tE))
    
def ddtE(tE, V, W):
    ret = 3*inner(tE, dot(grad(V), tE))*inner(tE, dot(grad(W), tE))*tE
    ret += - dot(grad(W), tE)*inner(tE, dot(grad(V), tE))
    ret += - dot(grad(V), tE)*inner(tE, dot(grad(W), tE))
    ret += - tE*inner(dot(grad(W), tE),dot(grad(V), tE))
    return ret

def dmu(mu, tE, V):
    #return dot(grad(V), mu) - mu*inner(mu, dot(grad(V), mu)) - tE*inner(tE, dot(grad(V)+ grad(V).T, mu))
    return dot( identity(3) - outer(mu, mu) - outer(tE, tE), dot( grad(V), mu)) - tE*inner(mu, dot(grad(V), tE))

def ddmu(mu, tE, V, W):
    ret = -dot(dot((identity(3)-outer(mu, mu)-outer(tE,tE)), dot(grad(W), dot( outer(mu, mu)+outer(tE, tE), grad(V)))), mu)
    ret +=-dot(dot((identity(3)-outer(mu, mu)-outer(tE,tE)), dot(grad(V), dot( outer(mu, mu)+outer(tE, tE), grad(W)))), mu)
    ret += -mu*inner(dot(grad(W), mu), dot(identity(3)-outer(mu, mu)-outer(tE,tE), dot(grad(V), mu)))
    ret += mu*inner(mu, dot(grad(W), tE))*inner(mu, dot(grad(V), tE))
    ret += -dot(identity(3)-outer(tE, tE), inner(mu, dot(grad(W), tE))*dot(grad(V),tE))
    ret += -dot(identity(3)-outer(tE, tE), inner(mu, dot(grad(V), tE))*dot(grad(W),tE))
    ret += tE*inner(dot(grad(W), tE), dot( outer(tE, tE)+outer(mu, mu), dot(grad(V).T, mu)))
    ret += tE*inner(dot(grad(V), tE), dot( outer(tE, tE)+outer(mu, mu), dot(grad(W).T, mu)))
    ret += -tE*inner(dot(grad(V), tE), dot(grad(W), mu))
    ret += tE*inner(tE, dot(grad(V), tE))*inner(tE, dot(grad(W), mu))
    ret += -tE*inner(dot(grad(W), tE), dot(grad(V), mu))
    ret += tE*inner(tE, dot(grad(W), tE))*inner(tE, dot(grad(V), mu))
    return ret

def divE(tE, V):
    return inner(tE, dot(grad(V), tE))('+')

def Hedge(tE, V, W):
    return -(divE(tE, V)*divE(tE, W)) + inner(dot(grad(V),tE), dot(grad(W), tE))('+')

def Hvol(V, W):
    return div(V)*div(W) - tr(dot(grad(V), grad(W)))

def Hsurf(n,V,W):
    return div(V)*div(W) - tr(dot(grad(V), grad(W))) + inner( dot(n, grad(V)), dot(n, grad(W)))
    
    
    
