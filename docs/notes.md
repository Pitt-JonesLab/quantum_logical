things to code up

1. flush out the simulator
   a. qudit channels
   b. a circuit representation with durations...

2. multi-optimization, qutrit gates with (ge, ef, gf) components

   - use our normal procedure, but optimize ge, ef into (0,0,0)
   - then optimize gf into desired gate
   - implement helper function for cartan trajectories?

3. revisit encodings? not sure exactly how to build with ooo in mind yet

later:

1. convert circuits into logical, insert in the EC protocol between gates

   - needs scheduling

2. variational circuit to learn encoding?
