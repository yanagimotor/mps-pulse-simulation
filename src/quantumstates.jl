using SparseArrays
using LinearAlgebra
using TensorOperations
using SpecialFunctions

include("./utils.jl")


"""
Generate a coherent state with mean field distribution αlist
"""
function coherent(αlist,n_l,χ,mmax)
    Γlist, λlist = emptymps_ΓΛ(n_l,χ,mmax)
    for Γi in 1:n_l
        Γlist[Γi] .= 0
        for mi in 1:mmax
            α = αlist[Γi]
            clist = [α^m/sqrt(Float64(factorial(big(m)))) for m in 0:mmax-1]
            clist ./= norm(clist)
            Γlist[Γi][mi,1,1] = clist[mi]
        end
    end
    for λi in 1:size(λlist)[1]
        λlist[λi].=0
        λlist[λi][1]=1.0
    end
    return Γlist, λlist
end

"""
Generate a coherent state for α under an envelope function flist
"""
function coherent(α,flist,n_l,χ,mmax)
    αlist=α*conj.(flist)
    return coherent(αlist,n_l,χ,mmax)
end

"""
Generate coherent state for three-wave mixing scenario
αlist and βlist specify signal and pump mean fields, respectively 
"""
function coherent_twm(αlist,βlist,n_l,χ,mmax)
    n_half=round(Int64,n_l/2)
    
    Γlist, λlist=emptymps_ΓΛ(n_l,χ,mmax)
    for Γi in 1:n_half
        odd_i=2*Γi-1
        even_i=2*Γi
        
        Γlist[odd_i].=0
        Γlist[even_i].=0
        
        for mi in 1:mmax
            α=αlist[odd_i]
            clist=[α^m/sqrt(Float64(factorial(big(m)))) for m in 0:mmax-1]
            clist./=norm(clist)
            Γlist[odd_i][mi,1,1]=clist[mi]
            
            β=βlist[even_i]
            clist=[β^m/sqrt(Float64(factorial(big(m)))) for m in 0:mmax-1]
            clist./=norm(clist)
            Γlist[even_i][mi,1,1]=clist[mi]
        end
    end
    for λi in 1:size(λlist)[1]
        λlist[λi].=0
        λlist[λi][1]=1.0
    end
    return Γlist, λlist
end


"""
Generate coherent state for three-wave mixing scenario
populate supermodes specified by envelope functions
"""
function coherent_twm(α,β,flist_α,flist_β,n_l,χ,mmax)
    αlist=conj.(flist_α)*α
    βlist=conj.(flist_β)*β
    return coherent_twm(αlist,βlist,n_l,χ,mmax)
end

"""
Generate a fock state ander an envelope flist
Bond dimension and Fock space cutoff are specified by χ_target and mmax_target, respectively
"""
function fock(n_fock,flist,χ_target,mmax_target;threshold=1e-10)
    mmax=n_fock+1
    χ=mmax
    n_l=length(flist)
    Mlist=emptymps_generic(n_l,χ,mmax)
    
    f_weight(si,k)=conj(flist[si])^k/sqrt(factorial(k))
    
    for si in 1:n_l
        if si==1
            for k in 0:n_fock
                Mlist[si][k+1,1,k+1]=f_weight(si,k)
            end
        elseif si==n_l
            for k in 0:n_fock
                Mlist[si][k+1,mmax-k,1]=f_weight(si,k)
            end
        else
            for k in 0:n_fock
                for α in 1:mmax-k
                    Mlist[si][k+1,α,α+k]=f_weight(si,k)
                end
            end
        end
    end
    
    Blist=right_normalize(Mlist)
    Γlist,λlist=ΓΛ_formalize(Blist,threshold=threshold)
    Γlist_t,λlist_t=reshape_dims(Γlist,λlist,χ_target,mmax_target)
    
    return Γlist_t,λlist_t
end