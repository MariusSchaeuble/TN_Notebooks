module DMRGHelperFunctions

using ITensors
using ITensorMPS

export build_left_blocks, build_right_blocks, get_heff, heff_apply, davidson, stretchBondDim


function build_left_blocks(psi::MPS, H::MPO)
    N = length(psi)
    L = Vector{ITensor}(undef, N)

    # Blocco iniziale: scalare 1
    L[1] = psi[1] * H[1] * dag(prime(psi[1]))

    for j in 2:N-1
        L[j] = L[j-1] * psi[j] * H[j] * dag(prime(psi[j]))
    end
    return L
end

function build_right_blocks(psi::MPS, H::MPO)
    N = length(psi)
    R = Vector{ITensor}(undef, N)

    R[N] = psi[N] * H[N] * dag(prime(psi[N]))

    for j in N-1:-1:2
        R[j] = R[j+1] * psi[j] * H[j] * dag(prime(psi[j]))
    end
    return R
end

function get_heff(h::MPO,k::Int, L::Vector{ITensor}, R::Vector{ITensor})
    if k == 1
        return R[k] * h[k]
    elseif k == length(h)
        return L[k-1] * h[k]
    else
        return L[k-1] * h[k] * R[k]
    end
end

function heff_apply(v::ITensor, k::Int, H::MPO,
                    L::Vector{ITensor}, R::Vector{ITensor})
    N = length(H)
    Hv = v * H[k]
    if k > 1
        Hv = Hv * L[k-1]
    end
    if k < N
        Hv = Hv * R[k+1]
    end
    return noprime(Hv)
end


"""
    davidson(apply_H, v0; maxiter, tol) -> (E, v)

Metodo di Davidson per trovare il ground state di H_eff.
apply_H: funzione che applica H_eff al tensore v.
"""
# function davidson(apply_H, v0::ITensor; maxiter=10, tol=1e-10)
#     # Rappresentiamo lo spazio di Krylov
#     V = [v0 / norm(v0)]
#     Hv = apply_H(V[1])

#     E_old = real(scalar(dag(V[1]) * Hv))
#     E = E_old
#     v = V[1]

#     for iter in 1:maxiter
#         # Residuo
#         r = Hv - E * V[end]
#         res_norm = norm(r)

#         if res_norm < tol
#             break
#         end

#         # Ortogonalizza il residuo rispetto allo spazio corrente
#         for vi in V
#             r = r - scalar(dag(vi) * r) * vi
#         end
#         r_norm = norm(r)
#         if r_norm < 1e-14
#             break
#         end
#         push!(V, r / r_norm)

#         # Costruisci la matrice di proiezione
#         dim = length(V)
#         M = zeros(ComplexF64, dim, dim)
#         HV = [apply_H(vi) for vi in V]
#         for i in 1:dim, j in 1:dim
#             M[i,j] = scalar(dag(V[i]) * HV[j])
#         end
#         M = (M + M') / 2  # Hermitianizza

#         # Diagonalizza
#         evals, evecs = eigen(M)
#         idx = argmin(real.(evals))
#         E = real(evals[idx])
#         c = evecs[:, idx]

#         # Aggiorna il vettore soluzione
#         v = sum(c[i] * V[i] for i in 1:dim)
#         v = v / norm(v)
#         Hv = apply_H(v)

#         if abs(E - E_old) < tol
#             break
#         end
#         E_old = E
#     end

#     return E, v
# end


function davidson(apply_H, v0::ITensor; maxiter=10, tol=1e-10)

    v = v0 / norm(v0)
    V  = ITensor[]
    HV = ITensor[]

    push!(V, v)
    push!(HV, apply_H(v))

    E = real(scalar(dag(v) * HV[1]))
    E_old = E

    for iter in 1:maxiter

        # Residuo sul vettore corrente
        Hv = apply_H(v)
        r = Hv - E * v
        res_norm = norm(r)

        if res_norm < tol
            break
        end

        # Ortogonalizza il residuo rispetto allo spazio corrente
        for vi in V
            r = r - scalar(dag(vi) * r) * vi
        end
        r_norm = norm(r)
        if r_norm < 1e-14
            break
        end

        # Aggiungi nuovo vettore allo spazio di Krylov
        new_v = r / r_norm
        push!(V, new_v)
        push!(HV, apply_H(new_v))  # cacha H|new_v>

        # Costruisci matrice di proiezione (solo nuove righe/colonne)
        dim = length(V)
        M = zeros(ComplexF64, dim, dim)
        for i in 1:dim, j in 1:dim
            M[i,j] = scalar(dag(V[i]) * HV[j])
        end
        M = (M + M') / 2

        # Diagonalizza e prendi autovalore minimo
        evals, evecs = eigen(M)
        idx = argmin(real.(evals))
        E = real(evals[idx])
        c = evecs[:, idx]

        # Aggiorna vettore corrente
        v = sum(c[i] * V[i] for i in 1:dim)
        v = v / norm(v)

        # Controlla convergenza
        if abs(E - E_old) < tol && res_norm < tol
            break
        end
        E_old = E
    end

    return E, v
end

function stretchBondDim(state::MPS,extDim::Int64)
   psiExt = copy(state);
   NN = length(psiExt)
   for n in 1:NN-1
      a = commonind(psiExt[n],psiExt[n+1])
      tagsa = tags(a)
      add_indx = Index(extDim, tags= tagsa)
      psiExt[n]=psiExt[n]*delta(a,add_indx)
      psiExt[n+1]=psiExt[n+1]*delta(a,add_indx)
   end
   #println("Overlap <original|extended> states: ", dot(state,psiExt));
   return psiExt, dot(state,psiExt)
end
end # module