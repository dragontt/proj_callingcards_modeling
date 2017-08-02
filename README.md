# CallingCards Modeling

This project models the Calling Cards (CC) peaks as the predictors of differentially expressed (DE) or not DE targets. There are two modeling approaches. (1) Binned promoter: assign CC features (e.g. TPH, RPH, distance to ATG) to the binned promoter, and use feature selection algorithms embeding logisitic regression to rank the features. (2) Tree based modeling: hierarchically cluster the transpositions into clusters (peaks), use the first few highest peaks for each gene target to build a decision tree or Random Forest, and rank the features by their importances output by the tree-based model. 

### Package Requirement

1. Install PyYAML
    
    ```
    pip install --user pyyaml
    ```

2. Install Bayesian Optimization by fmfn (git)

    ```
    pip install --user bayesian-optimization
    ```

3. Install mlxtend by rasbt (git)

    ```
    pip install --user mlxtend  
    ```

### Example Usage

1. Load modules

	```
	module load pandas
	module load scipy
	module load scikit-learn
    module load matplotlib
	module load biopython
    ```

2. Map transposition data to gene targets, whose promoters range from (-)1000bp upstream to (+)100bp downstream from the ATG. (Input: three-column gnashy files)

	```
	cd scripts
	python compute_orf_hops.py -r ../resources/ -pu -1000 -pd 100 -o ../output/
	```

3. Calculate Poisson statistics of the transpoisitions within the defined promoter region
    ```
    python find_sig_promoters.py -op ../output/ -gp ../resources/
    ``` 

4. Call peaks, allowing the maximum within cluster distance to be 200

	```
	python call_peaks.py -d ../output/ -c 200
	``` 

5. Create feature matrix for modeling peaks (for Approach (2))

	```
	python generate_features.py -m highest_peaks -i ../output/ -o ../output/
	```

6. Fit tree-based model and visualize data
	
	```
    python fit_model.py -m holdout_feature_variation -t highest_peaks -c ../output/ -o ../resources/optimized_cc_subset.txt -f ../output/feature_holdout_analysis.6_mimic_cc
	```

## Visualizing Peaks in IGV

1. Convert bed to bedgraph

    ```
    awk '{printf "%s\t%d\t%d\t%2.3f\n" , $1,$2,$3,$5}' <peak_calling_file>.bed > <peak_calling_file>.bedgraph
    ```

2. Open IGV, load genome, or open the saved IGV session file (*.xml)

