using QuantumPropagators.Arnoldi
using QuantumPropagators.Newton
H = [0 0 0 -1im;0 0 -1im 0;0 1im 0 0;1im 0 0 0.]
ψ = [1.0+0.0im;0;0;0]
wrk = NewtonWrk(ψ;m_max = 10)
ψ1 = copy(ψ)
newton!(ψ1,H,.1,wrk;max_restarts=200)
println(ψ1)