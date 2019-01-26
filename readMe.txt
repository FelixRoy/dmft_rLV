

In order to get an example of the ligth (and thus not-converged) solutions using both solvers (with gamma=0 and gamma!=0), the following line should be run in a terminal, once in the directory where the files are:

python runMe.py

_____________________________________________________________________________________________

The different python files are:

	dmft_withoutGamma.py
		Implements DMFT numerical solution in the perfectly asymmetric case (gamma=0).
		Therefore doesn't need to compute response function at each iteration.

	dmft_withGamma.py
		Implements DMFT numerical solution with any symmetry parameter gamma.
		This one does compute the response function at each iteration.
		I use the response integration for each trajectories here, and not Novikov.

	plot_dmft.py
		Defines some plotting functions for the DMFT results. Ill-commented.

	dmft_stationary.py
		Implements the static cavity computations (used in the plots). Ill-commented.





