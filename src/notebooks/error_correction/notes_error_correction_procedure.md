1. finding the specific gates that will be involved in the circuit that will require error correction
-use the parameters similar to the setup of the weyl chamber but adjust it to execute for a qudit(qutrit specifically)
-test to see the smallest time using the hamiltonian setup to find the gate of interest based on the desired operation
-the math done to find the desired gate will be attached in a pdf upload

2. test the gate in a circuit like setup with trotterization pre-error correction protecol 
-set up the circuit 
-simulate via trotterization with the inclusion of amp and phase dampening channels
    -make sure the units of time and normalization are correct
-if time permits use the hamiltonian setup to find this instead of using trotterization
    -this will be significantly slower than trotterization

3. pre-error correction benchmarking
-compare how the two gates(the 5wave and the 3wave) compare to eachother 
    -the 3wave should obvoiusly be better pre-error correction because based on the results that we have from experimentation the 3wave gate takes about a 10th of the time that 
    5wave gate takes to execute
    -there are disadvantages to both of these setups 

4. include the error correction protecol 
-right after the execution of the gate see if you get the desired result when you act on the system with the stabilizer codes

5. post-error correction benchmarking 
-compare the two gates after error correction has taken place
    -this may not be an obvious answer because the 5wave gate does not leave the codespace where as the 3wave does 




