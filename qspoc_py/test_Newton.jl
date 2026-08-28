using QuantumPropagators.Arnoldi
using QuantumPropagators.Newton
using LinearAlgebra
H = [0 0 0 -1im;0 0 -1im 0;0 1im 0 0;1im 0 0 0.]
ψ = [1.0+0.0im;0;0;0]
n = 8
x = rand(n,n) .+ 1im .* rand(n,n)
H = x + conj(transpose(x))
y = rand(n,1) .+ 1im .* rand(n,1)
ψ = y / norm(y,2)
wrk = NewtonWrk(ψ;m_max = 10)
ψ1 = copy(ψ)
newton!(ψ1,H,.3,wrk;max_restarts=200)
println(ψ1)
println(wrk.a)
println(maximum(abs.(wrk.leja)))
println(maximum(abs.(wrk.a)))