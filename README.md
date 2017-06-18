# Code used for the paper "Using Bad Learners to find Good Configurations".

./Data/ -> Contains the Data used in the paper.

progressive_sampling.py -> Run Progressive Sampling

projective_sampling.py -> Run Projective Sampling

rank_based_sampling.py -> Run Rank-based Method

atri_results.txt -> Contains all the results after running [author's code](https://github.com/atrisarkar/ASE_extn) for projective sampling

# How to run
- First execute all the methods
```
python rank_based_sampling.py
python progressive_sampling.py
python projective_sampling.py
```
The script would store all the results as a pickle file in ./PickleLocker

- Run merge_pickle.py
```
python merge_pickle.py
```
The script would merge all the pickle files generated in the previous step and store the merged pickle file (merged.p) in ./Statistics

- Run run_stats.py from ./Statistics/.
```
python run_stats.py
```
This script would generate the Skott-Knott charts similar to charts shown in Figure 7 of the paper.