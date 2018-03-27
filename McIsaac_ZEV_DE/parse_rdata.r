load('all_expression.rdata')

# tfs <- read.table("../CCData_GSE54831/avail_ZEV_TFs.txt", header=T)
# for (i in seq(dim(tfs)[1])) {
#     tf <- tfs$Gene[i] 
#     expr_mtx <- expression_data[expression_data$TF == tf & expression_data$time == 15, c('gene', 'cleaned')]
#     expr_mtx <- cbind(yeast_ids$Systematic[match(expr_mtx$gene, yeast_ids$Gene)], expr_mtx)
#     colnames(expr_mtx)[1] <- "#gene_sys"
#     tf <- tfs$Systematic[i]
#     write.table(expr_mtx, file=paste0("all_expression/",tf,"-15min.DE.txt"), quote=F, sep="\t", row.names=F)
# }

tfs <- unique(expression_data$TF)
for (tf in tfs) {
	## expression data of each TF induction
    expr_mtx <- expression_data[expression_data$TF == tf & expression_data$time == 15, c('gene', 'cleaned', 'strain')]
    ## make sure there are data at this timepoint
    if (dim(expr_mtx)[1] == 0) {
    	cat("Skipping ", tf, " <- no data at this timepoint\n")
    	next
    }

    cat("... working on\t", tf)
    ## choose one of the strains
    strains <- unique(expr_mtx$strain)
    expr_mtx <- expr_mtx[expr_mtx$strain == strains[1], c('gene', 'cleaned')]
    ## combine systematic and symbol names
    expr_mtx <- cbind(yeast_ids$Systematic[match(expr_mtx$gene, yeast_ids$Gene)], expr_mtx)
    colnames(expr_mtx)[1] <- "#gene_sys"
    tf_systematic <- yeast_ids$Systematic[yeast_ids$Gene == tf]
    cat("\t", tf_systematic, "\n")
    ## write output
    write.table(expr_mtx, file=paste0("all_expression/",tf_systematic,"-15min.DE.txt"), quote=F, sep="\t", row.names=F)
}

