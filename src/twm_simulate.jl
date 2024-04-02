using SparseArrays
using LinearAlgebra
using TensorOperations
using JLD

include("./twm.jl")


function simulate_twm(param_set,savepath)
    L=param_set["L"]
    n_l=param_set["n_l"]
    n_half=param_set["n_half"]
    mmax=param_set["mmax"]
    χ=param_set["chi"]
    J=param_set["J"]
    V=param_set["V"]
    β=param_set["beta"]
    Δ=param_set["delta"]
    α_a=param_set["alpha_a"]
    α_b=param_set["alpha_b"]
    flist_a=param_set["flist_a"]
    flist_b=param_set["flist_b"]
    nt=param_set["nt"]
    saveat=param_set["saveat"]
    th=param_set["th"]
    tmax=param_set["tmax"]
    
    Γlist,λlist=coherent_twm(α_a,α_b,flist_a,flist_b,n_l,χ,mmax);
    Γset,λset,norm_list=simulate_twm!(Γlist,λlist,J,β,V,Δ,tmax,nt,saveat;threshold=th);
    
    αset_a=[[mean(Γset[si],λset[si],ci) for ci in 1:2:n_l] for si in eachindex(saveat)]
    αset_b=[[mean(Γset[si],λset[si],ci) for ci in 2:2:n_l] for si in eachindex(saveat)]
    g1set_a=[[g1(Γset[si],λset[si],ci) for ci in 1:2:n_l] for si in eachindex(saveat)]
    g1set_b=[[g1(Γset[si],λset[si],ci) for ci in 2:2:n_l] for si in eachindex(saveat)]
    
    ρset=Array{Array{Complex{Float64},4}}(undef,length(saveat))
    
    Threads.@threads for si in eachindex(saveat)
        @time Γlist_t,λlist_t=sweepout(Γset[si],λset[si],[flist_a,flist_b];threshold=th);
        ρ=local_projection_left(Γlist_t,λlist_t,2);
        ρset[si]=ρ
    end
    
    save(savepath,"param_set",param_set,"alphaset_a",αset_a,"alphaset_b",αset_b,"g1set_a",g1set_a,"g1set_b",g1set_b,"rhoset",ρset,"Gammaset",Γset,"lambdaset",λset,"norm_list",norm_list)
end