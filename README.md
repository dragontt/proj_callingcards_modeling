# CallingCards Modeling

This project models the Calling Cards (CC) peaks as the predictors of differentially expressed (DE) or not DE targets. There are two modeling approaches. (1) Binned promoter: assign CC features (e.g. TPH, RPH, distance to ATG) to the binned promoter, and use feature selection algorithms embeding logisitic regression to rank the features. (2) Tree based modeling: hierarchically cluster the transpositions into clusters (peaks), use the first few highest peaks for each gene target to build a decision tree or Random Forest, and rank the features by their importances output by the tree-based model. 

### Example Usage

1. Map transposition data to gene targets, whose promoters range from (-)1000bp upstream to (+)100bp downstream from the ATG. (Input: three-column gnashy files)

    ```
    cd scripts
    python compute_orf_hops.py -r ../resources/ -u -1000 -d 100 -o ../output/
    ```

2. 
