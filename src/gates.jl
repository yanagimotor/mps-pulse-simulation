using SparseArrays
using LinearAlgebra
using TensorOperations
using SpecialFunctions

include("./utils.jl")

"""
Apply two-bit operation Vmat on a quantum state Γlist, λlist at index ci and ci+1
Elements of Vmat[i,j,k,l] are such that V=Vmat[i,j,k,l]|i,j><k,l|
"""
function twobit_operation!(Γlist,λlist,Vmat,ci;threshold=1e-10)
    n_l,χ,mmax=system_dim(Γlist)
    
    Γc=Γlist[ci]
    Γd=Γlist[ci+1]
    λβ=λlist[ci]
    λα= ci==1 ? [(i==1 ? 1.0 : 0.0) for i in 1:χ] : λlist[ci-1]
    λγ= ci==n_l-1 ? [(i==1 ? 1.0 : 0.0) for i in 1:χ] : λlist[ci+1]

    αmax= ci==1 ? 1 : χ
    γmax= ci==n_l-1 ? 1 : χ
    
    Γ_left=reshape([α>αmax ? 0.0im : λα[α]*Γc[k,α,β]*λβ[β] for k in 1:mmax, α in 1:χ, β in 1:χ],(mmax,χ,χ))
    Γ_right=reshape([γ>γmax ? 0.0im : Γd[l,β,γ]*λγ[γ] for l in 1:mmax, β in 1:χ, γ in 1:χ],(mmax,χ,χ))
    @tensor Γset[k,l,α,γ] :=Γ_left[k,α,β]*Γ_right[l,β,γ]
    @tensor Π[i,α,j,γ]:=Γset[k,l,α,γ]*Vmat[i,j,k,l] 

    F=SVD{ComplexF64, Float64, Matrix{ComplexF64}}
    try
        F = svd(reshape(Π,(mmax*χ,mmax*χ)),alg=LinearAlgebra.DivideAndConquer())
    catch e
        println(e)
        F = svd(reshape(Π,(mmax*χ,mmax*χ)),alg=LinearAlgebra.QRIteration())
    end
    
    λlist[ci]=[F.S[β]<threshold ? 0.0 : F.S[β] for β in 1:χ]
    U=reshape(F.U[:,1:χ],(mmax,χ,χ))
    Vt=reshape(F.Vt[1:χ,:],(χ,mmax,χ))
    
    for i in 1:mmax, α in 1:αmax, β in 1:χ
        Γlist[ci][i,α,β]= λα[α]<threshold ? 0.0im : U[i,α,β]/λα[α]
    end
    
    for j in 1:mmax, β in 1:χ, γ in 1:γmax
        Γlist[ci+1][j,β,γ]= λγ[γ]<threshold ? 0.0im : Vt[β,j,γ]/λγ[γ]
    end
    
end

"""
Apply one-bit operation Vmat on mode ci
V=Vmat[i,j]|i><j|
"""
function onebit_operation!(Γlist,Vmat,ci)
    n_l,χ,mmax=system_dim(Γlist)
    αmax= ci==1 ? 1 : χ
    βmax= ci==n_l ? 1 : χ
    
    Γlist[ci].=reshape([sum([Vmat[i,j]*Γlist[ci][j,α,β] for j in 1:mmax]) for i in 1:mmax, α in 1:αmax, β in 1:βmax],(mmax,αmax,βmax))
end


"""
Generate a matrix corresponding to a single-mode annihilation operator
"""
function destroy_op(mmax)
    mat=zeros(ComplexF64,mmax,mmax)
    for i in 1:mmax-1
        mat[i,i+1]=sqrt(i)
    end
    return mat
end

"""
Unitary for a single-mode Kerr nonlinearity + self-interaction from dispersion
(Exact solution)
"""
function onebit_unitary(U,V,δt,mmax)
    umat=zeros(ComplexF64,mmax,mmax)
    for n1 in 0:mmax-1
        ekerr=U/2*n1*(n1-1)
        elin=V*n1
        umat[n1+1,n1+1]=exp(-im*δt*(ekerr+elin))
    end
    return umat
end

"""
Unitary for hopping-interaction from dispersion
(Exact solution)
"""
function hopping_unitary(J,δt,mmax)
    umat=zeros(ComplexF64,mmax,mmax,mmax,mmax)
    s=im*sin(δt*J)
    c=cos(δt*J)
    for n1 in 0:mmax-1
        for n2 in 0:mmax-1
            for m1 in 0:n1, m2 in 0:n2
                n1t=m1+m2
                n2t=n1+n2-m1-m2
                if n1t+1>mmax || n2t+1>mmax
                    continue
                else
                    coeff=binomial(n1,m1)*binomial(n2,m2)*c^(m1)*(-conj(s))^m2*s^(n1-m1)*c^(n2-m2)
                    coeff*=sqrt(factorial(n1t+0.0))
                    coeff/=sqrt(factorial(n1+0.0))
                    coeff*=sqrt(factorial(n2t+0.0))
                    coeff/=sqrt(factorial(n2+0.0))
                    umat[n1+1,n2+1,n1t+1,n2t+1]+=coeff
                end
            end
        end
    end
    return umat
end

"""
Generate matrix representation of annihilation operators for a two-mode space
"""
function fieldoperators(fsize)
    destroy_mat=destroy_op(fsize)
    create_mat=transpose(destroy_op(fsize))
    id=destroy_mat*0.0im+I

    @tensor a[i1,i2,o1,o2]:=destroy_mat[i1,o1]*id[i2,o2]
    @tensor ad[i1,i2,o1,o2]:=create_mat[i1,o1]*id[i2,o2]
    @tensor b[i1,i2,o1,o2]:=destroy_mat[i2,o2]*id[i1,o1]
    @tensor bd[i1,i2,o1,o2]:=create_mat[i2,o2]*id[i1,o1]
    
    a=reshape(a,(fsize*fsize,fsize*fsize))
    ad=reshape(ad,(fsize*fsize,fsize*fsize))
    b=reshape(b,(fsize*fsize,fsize*fsize))
    bd=reshape(bd,(fsize*fsize,fsize*fsize))
    
    return a,ad,b,bd
end

"""
Unitary for three-wave mixing (first mode is the signal and the second mode is the pump) + self-interaction from dispersion
(fsize determins the numerical cutoff for Fock space)
"""
function twm_unitary(J,β,V,δt,mmax,fsize)
    a,ad,b,bd=fieldoperators(fsize)
    
    H=2*J*ad*a+2*β*J*bd*b+V/2*(ad^2*b+a^2*bd)
    umat=exp(-im*δt*H)
    
    return reshape(umat,(fsize,fsize,fsize,fsize))[1:mmax,1:mmax,1:mmax,1:mmax]
end


"""
Unitary for Kerr + self-interaction from dispersion
(fsize determins the numerical cutoff for Fock space)
"""
function ll_unitary(U,J,δt,mmax,fsize)
    a,ad,b,bd=fieldoperators(fsize)
    
    nl=U/4*ad*ad*a*a+J*ad*a+U/4*bd*bd*b*b+J*bd*b-J*(ad*b+a*bd)
    umat=exp(-im*δt*nl)
    
    return reshape(umat,(fsize,fsize,fsize,fsize))[1:mmax,1:mmax,1:mmax,1:mmax]
end

"""
Unitary for dissipative Kerr soliton simulation
(fsize determins the numerical cutoff for Fock space)
"""
function dks_unitary(U,J,D,R,δt,mmax,fsize)
    a,ad,b,bd=fieldoperators(fsize)
    
    nl=U/4*ad*ad*a*a+J*ad*a+U/4*bd*bd*b*b+J*bd*b-J*(ad*b+a*bd)
    hdisp=-D/2*(a+ad+b+bd)
    hparam=-R/4*(a*a+ad*ad+b*b+bd*bd)
    htot=nl+hdisp+hparam
    umat=exp(-im*δt*htot)
    
    return reshape(umat,(fsize,fsize,fsize,fsize))[1:mmax,1:mmax,1:mmax,1:mmax]
end

"""
Unitary operation to linearly mix two modes
(exact solution)
"""
function mix_unitary(ϕ,θ,mmax)
    umat=zeros(ComplexF64,mmax,mmax,mmax,mmax)
    s=exp(im*θ)*sin(ϕ)
    c=cos(ϕ)
    for n1 in 0:mmax-1
        for n2 in 0:mmax-1
            for m1 in 0:n1, m2 in 0:n2
                n1t=m1+m2
                n2t=n1+n2-m1-m2
                if n1t+1>mmax || n2t+1>mmax
                    continue
                else
                    coeff=binomial(n1,m1)*binomial(n2,m2)*s^m1*(-c)^m2*c^(n1-m1)*conj(s)^(n2-m2)
                    coeff*=sqrt(factorial(n1t+0.0))
                    coeff/=sqrt(factorial(n1+0.0))
                    coeff*=sqrt(factorial(n2t+0.0))
                    coeff/=sqrt(factorial(n2+0.0))
                    umat[n1+1,n2+1,n1t+1,n2t+1]+=coeff
                end
            end
        end
    end
    return umat
end

"""
Unitary for swapping two neighboring modes
(Exact solution)
"""
function swap_unitary(mmax)
    Smat=zeros(ComplexF64,mmax,mmax,mmax,mmax)
    for i in 1:mmax,j in 1:mmax
        Smat[i,j,j,i]=1.0
    end
    return Smat
end


"""
Usual beamsplitter operation to mix two modes
"""
function bs_unitary(ϕ,θ,mmax)
    umat=zeros(ComplexF64,mmax,mmax,mmax,mmax)
    s=exp(im*θ)*sin(ϕ)
    c=cos(ϕ)
    
    for n1 in 0:mmax-1
        for n2 in 0:mmax-1
            for m1 in 0:n1, m2 in 0:n2
                n1t=m1+m2
                n2t=n1+n2-m1-m2
                if n1t+1>mmax || n2t+1>mmax
                    continue
                else
                    coeff=binomial(n1,m1)*binomial(n2,m2)*c^m1*(-conj(s))^m2*(s)^(n1-m1)*c^(n2-m2)
                    coeff*=sqrt(factorial(n1t+0.0))
                    coeff/=sqrt(factorial(n1+0.0))
                    coeff*=sqrt(factorial(n2t+0.0))
                    coeff/=sqrt(factorial(n2+0.0))
                    umat[n1+1,n2+1,n1t+1,n2t+1]+=coeff
                end
            end
        end
    end
    return umat
end

"""
Unitary for single-mode squeezing
(fsize determins the numerical cutoff for Fock space)
"""
function squeezing(ξ,mmax,fsize)
    a=destroy_op(mmax)
    ad=adjoint(a)

    umat=exp(1/2*(ξ*ad*ad-conj(ξ)*a*a))
    
    return umat[1:mmax,1:mmax]
end
