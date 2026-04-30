# Parctice 3: Instructions

In today's practice we explore the definition and use of MPOs in ITensor. We will first have a very short look at how a given operator can be turned into an MPO starting from:

- the operator matrix: see `matrixProductOperators.ipynb`
- the bulk matrix: see `MPO_fromBulk.ipynb`

We then focus on an MPO construction completely based on `ITensorMPS` API; this is done in `MPO_ITensor.ipynb`.

In all the notebooks we use the evolution of an initial state under the action of the $XY$ Hamiltonian 

$$
H = -J \sum_{i=1}^{N-1} \sigma_-^{i} \sigma_+^{i+1} + \sigma_+^{i} \sigma_-^{i+1} 
$$ 

as extended example. This Hamiltonian admits exact solution and allows us, therefore, to check the correcness of our results. 

The exercises that you are "strongly encouraged" to complete within the next week are in `exercises_week3.ipynb`.





