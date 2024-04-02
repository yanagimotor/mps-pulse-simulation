using SparseArrays
using LinearAlgebra
using TensorOperations

include("./utils.jl")
include("./gates.jl")
include("./tomography.jl")
include("./quantumstates.jl")

# 01/18/2021: Bug fixed for simulate_twm!. This bug has not affected simulations with zero phase-mismatch

"""
Proceed three-wave mixing simulation
"""
function proceed_twm!(Γlist,λlist,Vmat_a,Vmat_b,Vmat_nl,Rmat,Smat;threshold=1e-10,forward=true)
    n_l,χ,mmax=system_dim(Γlist)
    c_nl_f=[j for j in 1:4:n_l-1]
    c_nl_s=[j for j in 3:4:n_l-1]
    
    c_sw1=[j for j in 1:4:n_l-1]
    
    c_a1=[j for j in 2:4:n_l-1]
    c_b1=[j for j in 4:4:n_l-1]
    
    c_sw2_f=[j for j in 1:4:n_l-1]
    c_sw2_s=[j for j in 3:4:n_l-1]
    
    c_b2=[j for j in 2:4:n_l-1]
    c_a2=[j for j in 4:4:n_l-1]
    
    c_sw3=[j for j in 3:4:n_l-1]
    
    cm_set=[(c_nl_f,Vmat_nl),(c_nl_s,Vmat_nl),(c_sw1,Smat),(c_a1,Vmat_a),(c_b1,Vmat_b),(c_sw2_f,Smat),
        (c_sw2_s,Smat),(c_b2,Vmat_b),(c_a2,Vmat_a),(c_sw3,Smat)]


    c_rmat=[j for j in 2:2:n_l]
    
    if forward
        for ci in c_rmat
            onebit_operation!(Γlist,Rmat,ci)
        end
        
        for cind in 1:length(cm_set)
            cilist,Vmat=cm_set[cind]
            Threads.@threads for ci in cilist
                twobit_operation!(Γlist,λlist,Vmat,ci,threshold=threshold)
            end
        end
    else
        for cind in length(cm_set):-1:1
            cilist,Vmat=cm_set[cind]
            Threads.@threads for ci in cilist[end:-1:1]
                twobit_operation!(Γlist,λlist,Vmat,ci,threshold=threshold)
            end
        end
        for ci in c_rmat[end:-1:1]
            onebit_operation!(Γlist,Rmat,ci)
        end
    end
end

"""
Simulate the dynamics under three-wave mixing interaction
J: signal dispersion
β*J: pump dispersion
Δ: pump detuning relative to signal
V: stregth of the three-wave mixing process
tmax: simulation time
nt: number of time steps
saveat: return values are quantum states at these time steps
"""
function simulate_twm!(Γlist,λlist,J,β,V,Δ,tmax,nt,saveat;threshold=1e-10)
    n_l,χ,mmax=system_dim(Γlist)
    δt=tmax/nt
    
    alist=[1/2]
    flist=[true]
    
    
    Vmats_a=Array{Any}(undef,length(alist))
    Vmats_b=Array{Any}(undef,length(alist))
    Vmats_nl=Array{Any}(undef,length(alist))
    Smat=swap_unitary(mmax)
    Rmats=Array{Any}(undef,length(alist))
    
    Γset=Array{Any}(undef,length(saveat))
    λset=Array{Any}(undef,length(saveat))
    norm_list=Vector{Float64}(undef,length(saveat))

    for (ai,a) in enumerate(alist)        
        Vmats_nl[ai]=twm_unitary(0,0,V,δt*a,mmax,2*mmax)
        Vmats_a[ai]=ll_unitary(0,J,a*δt,mmax,2*mmax)
        Vmats_b[ai]=ll_unitary(0,β*J,a*δt,mmax,2*mmax)
        Rmats[ai]=onebit_unitary(0,Δ,δt*a,mmax)
    end
   
    for ni in 1:nt
        println(ni)
        for ai in 1:length(alist)
            @time proceed_twm!(Γlist,λlist,Vmats_a[ai],Vmats_b[ai],Vmats_nl[ai],Rmats[ai],Smat;threshold=threshold,forward=flist[ai])       
        end
        for ai in length(alist):-1:1
           @time proceed_twm!(Γlist,λlist,Vmats_a[ai],Vmats_b[ai],Vmats_nl[ai],Rmats[ai],Smat;threshold=threshold,forward=!flist[ai]) 
        end
        
        for index in findall(x->x==ni,saveat)
            Γset[index]=[copy(Γ) for Γ in Γlist]
            λset[index]=[copy(λ) for λ in λlist]
            norm_list[index]=abs(overlap(Γlist,λlist,Γlist,λlist))
            
        end
    end
    
    return Γset,λset,norm_list
end

"""
Returns the signal waveform for simulton with average signal photon number n_ave
Pump envelope has half the displacement
"""
function simulton_waveform(n_ave,n_half,L)
    Δz=L/n_half
    zlist=[z for z in range(-L/2,stop=L/2,length=n_half)]
    ϕ0=(3*n_ave^2/32)^(1/3)
    k=sqrt(ϕ0/6)
    flist=[1/cosh(k*z)^2+0.0im for z in zlist]
    flist./=norm(flist)
    return flist
end