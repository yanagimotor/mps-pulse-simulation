using SparseArrays
using LinearAlgebra
using TensorOperations

"""
Extract system dimensions from a given quantum state representation
Γlist is a list of Γ-tensors with length n_l
First dimension of Γ is for mmax Fock space
Second and third dimensions of Γ are for bond dimensions

Return values: n_l, χ, mmax

n_l: Number of lattice sites
χ: Bond dimension
mmax: Number of local states
"""
function system_dim(Γlist)
    n_l=length(Γlist)
    χ=size(Γlist[1])[3]
    mmax=size(Γlist[1])[1]
    
    return n_l,χ,mmax
end

"""
Generate zero tensors for given MPS dimensions in a generic form
"""
function emptymps_generic(n_l,χ,mmax)
    Mlist=[ci==1 ? zeros(ComplexF64,mmax,1,χ) : (ci==n_l ? zeros(ComplexF64,mmax,χ,1) : zeros(ComplexF64,mmax,χ,χ)) for ci in 1:n_l]
    return Mlist
end

"""
Generate zero tensors for given MPS dimensions in a ΓΛ-form
"""
function emptymps_ΓΛ(n_l,χ,mmax)
    λlist=[zeros(Float64,χ) for i in 1:n_l-1]
    Γlist=[i==1 ? zeros(ComplexF64,mmax,1,χ) : (i==n_l ? zeros(ComplexF64,mmax,χ,1) : zeros(ComplexF64,mmax,χ,χ)) for i in 1:n_l]
    return Γlist,λlist
end

"""
Generate zero tensors of an MPO
"""
function emptympo(n_l,χ,mmax)
    Olist=[ci==1 ? zeros(ComplexF64,mmax,mmax,1,χ) : (ci==n_l ? zeros(ComplexF64,mmax,mmax,χ,1) : zeros(ComplexF64,mmax,mmax,χ,χ)) for ci in 1:n_l]
    return Olist
end

"""
Transform a given MPS with ΓΛ-form into a generic form
"""
function generic_form(Γlist,λlist)
    n_l,χ,mmax=system_dim(Γlist)    
    Mlist=emptymps_generic(n_l,χ,mmax)
    
    Mlist[1].=Γlist[1]
    for ci in 2:n_l
        for j in 1:mmax, α in 1:χ, β in 1:(ci==n_l ? 1 : χ)
            Mlist[ci][j,α,β]=Γlist[ci][j,α,β]*λlist[ci-1][α]
        end
    end
    return Mlist
end

"""
Transform a generic MPS into a right-normalized form
"""
function right_normalize(Mlist;normalize=true)
    n_l,χ,mmax=system_dim(Mlist)

    Blist=emptymps_generic(n_l,χ,mmax)
    
    usmat=zeros(ComplexF64,χ,χ)
    
    m_tilde=zeros(ComplexF64,mmax,χ,χ)
    
    for ci in n_l:-1:1
        if ci==n_l
            m_t=reshape([Mlist[ci][i,α,1] for α in 1:χ, i in 1:mmax],(χ,mmax))
            F=svd(m_t)
            
            if normalize
                F.S ./=norm(F.S)
            end
            
            usmat[axes(F.U,1),axes(F.U,2)].=F.U*diagm(F.S)
            
            Vt=permutedims(F.Vt,[2,1])
            Blist[ci][axes(Vt,1),axes(Vt,2),1].=Vt
            
        elseif ci==1
            m_t=zeros(ComplexF64,mmax,1,χ)
            for ki in 1:mmax
                m_t[ki,:,:].=view(Mlist[ci],ki,1:1,1:χ)*usmat
            end
            
            m_t=reshape([m_t[i,1,α] for i in 1:mmax, α in 1:χ],(1,mmax*χ))
            
            if normalize
                m_t./=norm(m_t)
            end
            
            Blist[ci].=reshape(m_t,(mmax,1,χ))
        else
            m_tilde.=0
            for ki in 1:mmax
                m_tilde[ki,:,:].=view(Mlist[ci],ki,1:χ,1:χ)*usmat
            end
            m_t=reshape(permutedims(m_tilde,[2,1,3]),(χ,mmax*χ))
            F=svd(m_t)
            
            if normalize
                F.S ./=norm(F.S)
            end
            
            usmat.=F.U*diagm(F.S)
            
            bmat=reshape(F.Vt,(χ,mmax,χ))
            Blist[ci].=reshape(permutedims(bmat,[2,1,3]),(mmax,χ,χ))
 
        end
        
    end
    
    return Blist
end

"""
Transform a generic MPS into a ΓΛ-normalized form
"""
function ΓΛ_formalize(Blist;threshold=1e-10)
    n_l,χ,mmax=system_dim(Blist)
    
    Γlist,λlist=emptymps_ΓΛ(n_l,χ,mmax)
    
    F=svd(reshape(Blist[1][:,1,:],(mmax,χ)))
    
    fv=size(F.U)[1]
    fh=size(F.U)[2]

    Γlist[1][1:fv,1:1,1:fh].=reshape(F.U,(fv,1,fh))
    λlist[1][1:length(F.S)].=[abs(sval)<threshold ? 0.0 : real(sval) for sval in F.S]
    
    b_prod=zeros(ComplexF64,mmax,χ,χ)
    b_tilde=zeros(ComplexF64,mmax*χ,χ)
    
    Vt=zeros(ComplexF64,χ,χ)
    Vt[axes(F.Vt,1),axes(F.Vt,2)].=F.Vt
    
    for ci in 2:n_l-1
        for i in 1:mmax
            b_prod[i,:,:].=diagm(λlist[ci-1])*Vt*Blist[ci][i,:,:]
        end

        b_tilde=reshape(b_prod,(mmax*χ,χ))
        F=svd(b_tilde)
        Vt.=F.Vt
        λlist[ci].=[abs(sval)<threshold ? 0.0 : real(sval) for sval in F.S]

        u=reshape(F.U,(mmax,χ,χ))
        Γlist[ci].=reshape([abs(λlist[ci-1][α])<threshold ? 0.0im : u[i,α,β]/λlist[ci-1][α] for i in 1:mmax, α in 1:χ, β in 1:χ],(mmax,χ,χ))
        
    end
    
    for i in 1:mmax
        Γlist[n_l][i,:,:].=F.Vt*Blist[n_l][i,:,:]
    end
    
    return Γlist,λlist
end

"""
Reshape the dimensionality of the MPS
"""
function reshape_dims(Γlist,λlist,χ_target,mmax_target)
    n_l,χ_current,mmax_current=system_dim(Γlist)
    
    Γlist_t,λlist_t=emptymps_ΓΛ(n_l,χ_target,mmax_target)
    
    mmax_set=min(mmax_current,mmax_target)
    χ_set=min(χ_current,χ_target)
    
    for ci in 1:n_l
        Γlist_t[ci][1:mmax_set,1:(ci==1 ? 1 : χ_set),1:(ci==n_l ? 1 : χ_set)].=Γlist[ci][1:mmax_set,1:(ci==1 ? 1 : χ_set),1:(ci==n_l ? 1 : χ_set)]
    end
    
    for ci in 1:n_l-1
        λlist_t[ci][1:χ_set].=λlist[ci][1:χ_set]
    end
    
    return Γlist_t,λlist_t
end