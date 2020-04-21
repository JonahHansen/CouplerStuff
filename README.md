# CouplerStuff
Random scripts for calculating coupler/fringe tracking things

Module files
* tricoupler: functions for tricoupler calculations
* fringe_functions: functions for calculating the intensity of fringes, as well as glass dispersion and group delay tracking

Scripts:
* plot_fringe_packet: Plots a fringe packet (as it says...)
* group_delay_sim: Uses phasors to estimate the group delay and then estimates the visibility by applying a delay correction.
* group_delay_sim_coupler: Same as group_delay_sim, but uses the tricoupler functions rather than the intensity equation from fringe_functions. Note that this does not use fake data and noise, and is more a conceptual test.
* group_delay_sim_loop: Basically group_delay_sim in a loop, seeing how the visibility changes over time for a given SNR. Need to fix atmospheric simulation and adjust incoherent integration components.
* group_delay_AC: Tries to find the group delay using only an AC coupler. Does not estimate
visibility. Does not work!
