using SparseArrays
using LinearAlgebra
using TensorOperations

"""
Perform a tensor contruction to obtain expectation value of an MPO
"""
function expect_value(Mlist,Olist;id_list=Vector{Int64}(undef,0))
    n_l=length(Mlist)
    αs=size(Mlist[1])[3]
    β=size(Olist[1])[4]

    mmax=size(Mlist[1])[1]
    
    mat1=Mlist[1][:,1,:]
    mat2=Olist[1][:,:,1,:]
    @tensor O[α,β,j]:=mat1[i,α]*mat2[j,i,β]

    mat=conj(Mlist[1][:,1,:])
    @tensor U1[α,β,γ]:=O[α,β,k]*mat[k,γ]

    
    for ci in 2:n_l-1
        if ci in id_list
            @tensor U1[α,β,γ]:=U1[μ,β,ν]*Mlist[ci][i,μ,α]*conj(Mlist[ci][i,ν,γ])
        else
            @tensor U2[α,β,γ,i]:=U1[μ,β,γ]*Mlist[ci][i,μ,α]
            @tensor U2[α,β,γ,i]:=U2[α,μ,γ,k]*Olist[ci][i,k,μ,β]
            @tensor U1[α,β,γ]:=U2[α,β,μ,k]*conj(Mlist[ci][k,μ,γ])
        end
    end
    
    mat=Mlist[n_l][:,:,1]
    @tensor W1[β,γ,i]:=U1[α,β,γ]*mat[i,α]
    mat=Olist[n_l][:,:,:,1]
    @tensor W2[γ,i]:=W1[β,γ,k]*mat[i,k,β]
    
    mat=conj(Mlist[n_l][:,:,1])
    @tensor W3[]:=W2[γ,i]*mat[i,γ]

    return W3[1]
end

"""
Compute a local expectation value of an operator vmat
Generally, faster than expect_value
"""
function expect_value_local(Γlist,λlist,ci,vmat)
    n_l,χ,mmax=system_dim(Γlist)
    iv=zeros(ComplexF64,mmax)
    jv=zeros(ComplexF64,mmax)
    
    Γc=Γlist[ci]
    λα= ci==1 ? [(i==1 ? 1.0 : 0.0) for i in 1:χ] : λlist[ci-1]
    λβ= ci==n_l ? [(i==1 ? 1.0 : 0.0) for i in 1:χ] : λlist[ci]
    
    su=0.0im
    for i in 1:mmax, j in 1:mmax, α in 1: (ci==1 ? 1 : χ), β in 1: (ci==n_l ? 1 : χ)
        su+=vmat[i,j]*abs(λα[α])^2*abs(λβ[β])^2*conj(Γc[i,α,β])*Γc[j,α,β]
    end
    return su
end

"""
Calculate g1-function, i.e., photon number distribution
"""
function g1(Γlist,λlist,ci)
    n_l,χ,mmax=system_dim(Γlist)
    vmat=diagm([j for j in 0:mmax-1])
    return expect_value_local(Γlist,λlist,ci,vmat)
end

"""
Calculate meanfield displacement
"""
function mean(Γlist,λlist,ci)
    n_l,χ,mmax=system_dim(Γlist)
    vmat=zeros(ComplexF64,mmax,mmax)
    for i in 1:mmax-1
        vmat[i,i+1]=sqrt(i)
    end
    return expect_value_local(Γlist,λlist,ci,vmat)
end

"""
Unconditional projection on n-photon state
"""
function nphoton_project(Γlist,λlist,n,ci)
    n_l,χ,mmax=system_dim(Γlist)
    vmat=zeros(ComplexF64,mmax,mmax)
    vmat[n+1,n+1]=1.0
    return expect_value_local(Γlist,λlist,ci,vmat)
end

"""
Unconditional photono number distribution
"""
function photondist(Γlist,λlist,ci)
    n_l,χ,mmax=system_dim(Γlist)   
    return [nphoton_project(Γlist,λlist,n,ci) for n in 0:mmax-1]
end

"""
Calculate two-photon correlation function
corr_2(ci1,ci2)/g1(ci1)/g1(ci2)=g2(ci1,ci2)
"""
function corr_2(Γlist,λlist,ci1,ci2)
    n_l,χ,mmax=system_dim(Γlist)
    Olist=emptympo(n_l,1,mmax)
    
    for cind in 1:n_l
        for m in 1:mmax
            if ci1==ci2
                Olist[cind][m,m,1,1]= cind==ci1 ? (m-1)*(m-2) : 1
            else
                Olist[cind][m,m,1,1]= (cind==ci1 || cind==ci2) ? (m-1) : 1
            end
        end
    end
    
    Mlist=generic_form(Γlist,λlist)
    return expect_value(Mlist,Olist)
end


"""
Calculate two-point field correlation function
<a^dagger_{ci1} a_{ci2}>
"""
function twopoint_corr(Mlist,ci1,ci2)
    n_l,χ,mmax=system_dim(Mlist)
    Olist=emptympo(n_l,1,mmax)
    
    for cind in 1:n_l
        if cind == ci1 || cind == ci2
            if ci1==ci2   
                for m in 1:mmax
                    Olist[cind][m,m,1,1] = (m-1)
                end
            else
                for i in 1:mmax-1
                    if cind == ci2
                        Olist[cind][i,i+1,1,1]=sqrt(i)
                    else
                        Olist[cind][i+1,i,1,1]=sqrt(i)
                    end
                end
            end
        else
            for m in 1:mmax
                Olist[cind][m,m,1,1] = 1
            end
        end
        
    end
    
    return expect_value(Mlist,Olist;id_list=[ci for ci in 1:n_l if !(ci in [ci1,ci2])])
end

function twopoint_corr(Γlist,λlist,ci1,ci2)
    Mlist=generic_form(Γlist,λlist)
    return twopoint_corr(Mlist,ci1,ci2)
end

"""
Calculate two-point field correlation function
<a_{ci1} a_{ci2}>
"""
function twopoint_corr_annihilation(Mlist,ci1,ci2)
    n_l,χ,mmax=system_dim(Mlist)
    Olist=emptympo(n_l,1,mmax)
    
    for cind in 1:n_l
        if cind == ci1 || cind == ci2
            if ci1==ci2   
                for m in 1:mmax-2
                    Olist[cind][m,m+2,1,1] = sqrt(m+1)*sqrt(m)
                end
            else
                for i in 1:mmax-1
                    Olist[cind][i,i+1,1,1]=sqrt(i)
                end
            end
        else
            for m in 1:mmax
                Olist[cind][m,m,1,1] = 1
            end
        end
        
    end
    
    return expect_value(Mlist,Olist;id_list=[ci for ci in 1:n_l if !(ci in [ci1,ci2])])
end

function twopoint_corr_annihilation(Γlist,λlist,ci1,ci2)
    
    Mlist=generic_form(Γlist,λlist)
    return twopoint_corr_annihilation(Mlist,ci1,ci2)
end

function sweepout_angles(fset,n_l)
    
    for si in 1:length(fset)
        for si2 in 1:1:si-1
            fset[si]-=(adjoint(fset[si2])*fset[si])*fset[si2]
        end
        fset[si]=fset[si]/norm(fset[si])
    end
    
    function trans_mat(θlist,φlist,θn,n,r)
        mat=zeros(ComplexF64,n,n)+I
        tmat=zeros(ComplexF64,n,n)+I
        
        mat[n,n]=exp(im*θn)
        for m in length(θlist):-1:1
            θ=θlist[m]
            φ=φlist[m]
            tmat.=0
            tmat+=I
            
            mt=[exp(im*θ)*sin(φ) cos(φ); -cos(φ) exp(-im*θ)*sin(φ)]

            mat[m+r-1:m+1+r-1,1:end]=mt*mat[m+r-1:m+1+r-1,1:end]
        end
        return mat
    end

    
    θset=Array{Array{Float64,1}}(undef,length(fset));
    φset=Array{Array{Float64,1}}(undef,length(fset));
    θnset=zeros(Float64,length(fset))
    cmat=zeros(ComplexF64,n_l,n_l)+I;

    for r in 1:length(fset)
        flist=fset[r]
        glist=zeros(ComplexF64,n_l-r+1)
        offset=(r-1)
        
        θset[r]=zeros(Float64,n_l-r)
        φset[r]=zeros(Float64,n_l-r)
        

        for l in 1:1:n_l-r+1
        
            f=flist[l]
            for m in r:1:r+l-2
                f-=glist[m-offset]*cmat[m,l]
            end
            f/=cmat[r+l-1,l]

            pr= (l==1) ? 1.0 : prod([cos(φset[r][k-offset]) for k in r:1:r+l-2])
            
            if l==n_l-r+1
                θnset[r]=angle(f/pr)
            else
                val=f/pr

                if abs(val)>1
                    println("WARNING Numerical error: val="*string(abs(val)))
                    val/=abs(val)
                end
                if l==n_l-r+1
                    φ=pi/2
                else
                    try
                        φ=asin(abs(val))
                    catch e
                        println("Numerical warning")
                        φ=pi/2
                    end
                end
                θ=angle(val*sign(sin(φ)))

                θset[r][r+l-1-offset]=θ
                φset[r][r+l-1-offset]=φ

                glist[l]=f
            end
        end
        
        t_mat=trans_mat(θset[r],φset[r],θnset[r],n_l,r)
        
        cmat=t_mat*cmat
    end

    
    return θset,φset,θnset,cmat
end

"""
For set of supermodes in f_set, calculate beamsplitter angles to realize sweep-out scheme
Supermodes are demultiplex to the leflmost bins
"""
function sweepout_angles_bigfloat(fset,n_l)
    
    fset=Vector{Complex{BigFloat}}.(fset)
    for si in 1:length(fset)
        for si2 in 1:1:si-1
            fset[si]-=(adjoint(fset[si2])*fset[si])*fset[si2]
        end
        fset[si]=fset[si]/norm(fset[si])
    end
    
    function trans_mat(θlist,φlist,θn,n,r)
        mat=zeros(Complex{BigFloat},n,n)+I
        tmat=zeros(Complex{BigFloat},n,n)+I
        
        mat[n,n]=exp(im*θn)
        for m in length(θlist):-1:1
            θ=θlist[m]
            φ=φlist[m]
            tmat.=0
            tmat+=I
            
            mt=[exp(im*θ)*sin(φ) cos(φ); -cos(φ) exp(-im*θ)*sin(φ)]

            mat[m+r-1:m+1+r-1,1:end]=mt*mat[m+r-1:m+1+r-1,1:end]
        end
        return mat
    end

    
    θset=Array{Array{BigFloat,1}}(undef,length(fset));
    φset=Array{Array{BigFloat,1}}(undef,length(fset));
    θnset=zeros(BigFloat,length(fset))
    cmat=zeros(Complex{BigFloat},n_l,n_l)+I;

    for r in 1:length(fset)
        flist=fset[r]
        glist=zeros(Complex{BigFloat},n_l-r+1)
        offset=(r-1)
        
        θset[r]=zeros(BigFloat,n_l-r)
        φset[r]=zeros(BigFloat,n_l-r)
        

        for l in 1:1:n_l-r+1
        
            f=flist[l]
            for m in r:1:r+l-2
                f-=glist[m-offset]*cmat[m,l]
            end
            f/=cmat[r+l-1,l]

            pr= (l==1) ? 1.0 : prod([cos(φset[r][k-offset]) for k in r:1:r+l-2])
            
            if l==n_l-r+1
                θnset[r]=angle(f/pr)
            else
                val=f/pr

                if abs(val)>1
                    println("WARNING Numerical error: val="*string(abs(val)))
                    val/=abs(val)
                end
                if l==n_l-r+1
                    φ=pi/2
                else
                    try
                        φ=asin(abs(val))
                    catch e
                        println("Numerical warning")
                        φ=pi/2
                    end
                end
                θ=angle(val*sign(sin(φ)))

                θset[r][r+l-1-offset]=θ
                φset[r][r+l-1-offset]=φ

                glist[l]=f
            end
        end
        
        t_mat=trans_mat(θset[r],φset[r],θnset[r],n_l,r)
        
        cmat=t_mat*cmat
    end
    
    θset_f=Vector{ComplexF64}.(θset)
    φset_f=Vector{ComplexF64}.(φset)
    θnset_f=Vector{ComplexF64}(θnset)
    cmat_f=Matrix{ComplexF64}(cmat)
    
    return θset_f,φset_f,θnset_f,cmat_f
end

"""
Perform sweep-out operation for a given set of supermodes
Return a newly produced state with given dimensionalities
"""
function sweepout(Γlist,λlist,fset;threshold=1e-10,χ_target=-1,mmax_target=-1,bigfloat=false)
    n_l,χ_c,mmax_c=system_dim(Γlist)
    
    mmax=mmax_target>0 ? mmax_target : mmax_c
    χ=χ_target>0 ? χ_target : χ_c
    
    if bigfloat
        θset,ϕset,θnset,cmat=sweepout_angles_bigfloat(fset,n_l)
    else
        θset,ϕset,θnset,cmat=sweepout_angles(fset,n_l)
    end

    Γlist_t,λlist_t=reshape_dims(Γlist,λlist,χ,mmax)

    for si in 1:length(fset)
        vmat=onebit_unitary(0.0,1.0,-θnset[si],mmax)
        onebit_operation!(Γlist_t,vmat,n_l)
        for ci in n_l-1:-1:si
            ϕ=ϕset[si][ci-(si-1)]
            θ=θset[si][ci-(si-1)]
            umat=mix_unitary(ϕ,θ,mmax)
            twobit_operation!(Γlist_t,λlist_t,umat,ci;threshold=1e-10)
        end
    end
    return Γlist_t, λlist_t
end


"""
Take partial trace of the system composed of mode s+1 to n_l
Reduced density matrix contains states of mode 1,2,…,s
"""
function local_projection_left(Γlist,λlist,s)
    function expand_index(n,inds,dim,mmax)
        for i in 1:dim
            si=n%mmax
            inds[i]=si+1
            n=round(Int64,(n-si)/mmax)
        end
    end
    
    Mlist=generic_form(Γlist,λlist)
    n_l,χ,mmax=system_dim(Γlist)
    
    ρ=zeros(ComplexF64,[mmax for _ in 1:2*s]...)
    
    @tensor U[α,β]:=Mlist[n_l][i,α,1]*conj(Mlist[n_l][i,β,1])
    
    for m in n_l-1:-1:s+1
        mat1=Mlist[m][:,:,:]
        mat2=conj.(Mlist[m][:,:,:])
        @tensor U[α,β]:=U[αp,βp]*mat1[i,α,αp]*mat2[i,β,βp]
    end

    inds1=zeros(Int64,s)
    inds2=zeros(Int64,s)
    
    for n1 in 0:mmax^(s)-1
        expand_index(n1,inds1,s,mmax)
        for n2 in 0:mmax^s-1
            expand_index(n2,inds2,s,mmax)
            
            V=copy(U)
            for m in s:-1:1
                mat1=Mlist[m][inds1[m],:,:]
                mat2=conj.(Mlist[m][inds2[m],:,:])
                @tensor V[αn,βn]:=V[αp,βp]*mat1[αn,αp]*mat2[βn,βp]
            end
            val=V[1,1]
            ρ[vcat(inds1,inds2)...]=val
        end
    end
    
    return ρ
end


"""
Calculate matrix elements of a displacement operator
"""
function disp_element(α,m,n)
    fact=exp(-abs(α)^2/2)
    fact*=sqrt(factorial(m+0.0))*sqrt(factorial(n+0.0))
    
    su=0.0
    for k in 0:min(m,n)
        su+=α^(m-k)*(-conj(α))^(n-k)/factorial(k+0.0)/factorial(m-k+0.0)/factorial(n-k+0.0)
    end
    return su*fact
end

"""
Calculate the Wigner function of a given single-mode density matrix
"""
function wigner_function(ρmat,xlist,ylist)
    mmax=size(ρmat)[1]
    wmap=zeros(Float64,length(xlist),length(ylist))
    xmap=zeros(Float64,length(xlist),length(ylist))
    ymap=zeros(Float64,length(xlist),length(ylist))

    for xi in eachindex(xlist), yi in eachindex(ylist)
        x=xlist[xi]
        y=ylist[yi]

        wsum=0.0im
        for m in 0:mmax-1, n in 0:mmax-1
            wsum+=(-1)^n*disp_element(sqrt(2)*(x+im*y),m,n)*ρmat[n+1,m+1]
        end

        wmap[xi,yi]=real(1/pi*wsum)
        xmap[xi,yi]=xlist[xi]
        ymap[xi,yi]=ylist[yi]
    end
    return xmap,ymap,wmap
end

"""
Compute the overlab between two MPS
No normalization is assumed
"""
function overlap(Γlist1,λlist1,Γlist2,λlist2)
    n_l,χ,mmax=system_dim(Γlist1)
    mat1=view(Γlist1[1],axes(Γlist1[1])...)
    mat2=view(Γlist2[1],axes(Γlist2[1])...)
    @tensor ua[α,β]:=mat1[i,1,α]*conj(mat2[i,1,β])
    
    for ci in 2:n_l
        for α in 1:χ
            ua[α,:].*=λlist1[ci-1][α]
            ua[:,α].*=λlist2[ci-1][α]
        end
        
        mat1=view(Γlist1[ci],axes(Γlist1[ci])...)
        @tensor ub[αn,β,i]:=mat1[i,αp,αn]*ua[αp,β]
        
        mat2=view(Γlist2[ci],axes(Γlist2[ci])...)
        @tensor ua[α,βn]:=ub[α,βp,i]*conj(mat2[i,βp,βn])
    end
    
    return ua[1,1]
end

"""
Matrix product operator that projects the state onto (i-1) Fock state on the ci th spatial bin
"""
function photon_project_operator(n_l,mmax,ci,i)
    Olist=emptympo(n_l,1,mmax)
    
    for cind in 1:n_l
        if cind==ci
            Olist[cind][i,i,1,1]=1.0
        else
            for mi in 1:mmax
                Olist[cind][mi,mi,1,1]=1.0
            end
        end
    end
    
    return Olist
end

"""
An identity matrix product operator
"""
function identity_operator(n_l,mmax)
    Olist=emptympo(n_l,1,mmax)
    for cind in 1:n_l
        for mi in 1:mmax
            Olist[cind][mi,mi,1,1]=1.0
        end
    end
    return Olist
end

"""
Take a local photon counting sample
"""
function sample_photoncount(Γlist,λlist)
    n_l,χ,mmax=system_dim(Γlist)

    Mlist=generic_form(Γlist,λlist);
    Olist_id=identity_operator(n_l,mmax);

    mind_list=zeros(Int64,n_l)
    for ci in 1:n_l
        nor=expect_value(Mlist,Olist_id)
        probs=[expect_value(Mlist,photon_project_operator(n_l,mmax,ci,i))/nor for i in 1:mmax]
        r=rand(Float64)

        mind=mmax
        s=0.0
        for mi in 1:mmax
            s+=real(probs[mi])
            if s>=r
                mind=mi
                break
            end
        end
        mind_list[ci]=mind


        Mlist[ci][1:1:mind-1,:,:].=0.0
        Mlist[ci][mind+1:1:end,:,:].=0.0
    end
    return mind_list.-1
end
