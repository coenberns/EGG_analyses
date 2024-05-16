This repo generally represents all data of importance for the analysis of the Electrogastrography (EGG) data recorded by MiGUT. 

More particularly, there are 3 main .py function files: 
- functions_read_bursts.py -> a function file containing all functions for read-in, data-prep, and analysis of the EGG data and various functions for validation of the new recording mode (also interpolation, smoothening of the signal etc.)
- Old_Plot_EGG.py -> All functions from previous data analyses, contributed by Sean (Siheng) You and Adam Gierlach
- Plot_EGG_adaptation.py -> The functions from Old_Plot_EGG.py including adaptations for specific use cases and figures

Then, there are several analysis files, utilizing the functions in the files above, to generate the results for the figures. 
These are typically markated by Analysis_XXXX_someinfo.py  where XXXX represents the date of the _in vivo_ recording. 
Various dataframes which were saved and can be used in CEBRA for the EGG x Behavioral characterization merged self-supervised and supervised learning models in my **Behavioral** repo. 

If there are any specific questions, mainly about the analysis or generation of figures, please contact me at coenjpberns@gmail.com
