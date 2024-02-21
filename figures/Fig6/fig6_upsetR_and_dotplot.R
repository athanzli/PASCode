library('UpSetR')
library(ggplot2)

setwd('/home/che82/athan/PASCode/code/github_repo/figures/fig6')

################################################################################
# setup
################################################################################
subclass_palette <- read.csv("/home/che82/athan/PASCode/code/github_repo/figures/subclass_palette.csv", header=FALSE)
subclass_palette <- setNames(subclass_palette$V2[2:dim(subclass_palette)[1]], subclass_palette$V1[2:dim(subclass_palette)[1]])
old_names <- names(subclass_palette)
subclass_palette <- setNames(subclass_palette, c(1:28))
names_with_space <- paste0(names(subclass_palette), " ")
new <- setNames(subclass_palette, names_with_space)
subclass_palette <- c(subclass_palette, new)
subclass_int_map <- setNames(c(1:28), old_names)

phenotypes <- c('AD', 'SleepWeightGainGuiltSuicide', 'WeightLossPMA', 'DepressionMood')

################################################################################
# upsetR plot
################################################################################
library(jsonlite)
options <- c('c02x', 'c90x', 'c91x', 'c92x')
up_down <- 'up' # TODO NOTE

listInput <- vector("list", length(options))
listInput <- setNames(vector("list", length(phenotypes)), phenotypes)

chosen_ctp = 'Oligo' # NOTE TODO
i <- 1
for (option in options){
  dic <- fromJSON(paste0('/home/che82/athan/PASCode/code/github_repo/figures/fig6/dic_', up_down, '_', option, '.json'))  
  listInput[phenotypes[i]] <- dic[chosen_ctp]
  i <- i + 1
}

pdf(paste0("/home/che82/athan/PASCode/code/github_repo/figures/fig6/upsetplot_", chosen_ctp, "_", up_down, ".pdf"),
    pointsize=12, width = 18, height=7)
upset(fromList(listInput), 
      nintersects = NA,
      sets = phenotypes,
      nsets = 4,
      sets.x.label = "Number of upregulated DEGs",
      mainbar.y.label = paste0("Number of common upregulated DEGs in ", chosen_ctp),
      point.size = 5,
      
      # Color Customizations
      matrix.color = "black",          # Change the color of the intersection points
      main.bar.color = "black",         # Change the color of the main bar plot
      sets.bar.color = "black",         # Change the color of the set size bar plot
      
      # Text and Font Customizations
      text.scale = 1.75,                   # Increase the overall text size (including axis labels and tick labels)
      set_size.numbers_size = 4.5,      # Increase the font size of the numbers on the set size bar chart
      
      # Log-scale Customizations
      scale.intersections = "log2", # TODO! just the bar scale not the number!
      # scale.sets = "log2"
      
      # Highlight "AD" related intersections
      # queries = list(
        # list(query = intersects, params = list("AD"), color = "purple", active = T),
        # list(query = intersects, params = list("AD", "WeightLossPMA"), color = "blue", activex = T)#,
        # list(query = intersects, params = list("AD", "SleepWeightGainGuiltSuicide"), color = "blue", active = T),
        # list(query = intersects, params = list("AD", "DepressionMood"), color = "blue", active = T)
      # )
)
dev.off()



################################################################################
# dot plot using Seurat DotPlot
################################################################################
# step 1: load data
gset <- unlist(read.csv('gset.csv', header=FALSE), use.names=FALSE)
avg_exp_list <- unlist(read.csv('avg_exp.csv', header=FALSE), use.names=FALSE)
pct_exp_list <- unlist(read.csv('pct_exp.csv', header=FALSE), use.names=FALSE)
data <- data.frame(
  Group = rep(paste0(phenotypes, ' PAC+'), each = length(gset)),
  Feature = rep(gset, times = length(phenotypes)),
  LogAvg.Expression = avg_exp_list,
  PercentExpressed = pct_exp_list
)

# step 2: plot
# pdf(paste0("/home/che82/athan/PASCode/code/github_repo/figures/fig6/gene_dotplot_", chosen_ctp, "_", up_down, ".pdf"), 
#     pointsize=12, width = 18, height=7)
ggplot(data, aes(x = Feature, y = Group)) +
  geom_point(aes(color = LogAvg.Expression, size = PercentExpressed)) +
  scale_color_gradient(low = "blue", high = "red") +
  scale_size(range = c(2, 10)) +
  theme_minimal() +
  labs(title = "", x = "", y = "") +
  theme(
    text = element_text(size=14),
    axis.text = element_text(size=17.8),
    axis.text.x = element_text(angle=45, hjust=1)
  )
# dev.off(


# ################################################################################
# # dot plot using Seurat DotPlot
# ################################################################################
# # step 1: choose genes
# ## according to upsetR plot
# l1 = unlist(listInput['AD'], use.names=FALSE)
# l2 = unlist(listInput['SleepWeightGainGuiltSuicide'], use.names=FALSE)
# l3 = unlist(listInput['WeightLossPMA'], use.names=FALSE)
# l4 = unlist(listInput['DepressionMood'], use.names=FALSE)
# gset1 = intersect(l1, l2)
# gset2 = intersect(l1, l3)
# gset3 = intersect(l1, l4)
# gset4 = intersect(intersect(l3, l4), l2)
# gene_features <- union(union(gset1, gset2), union(gset3, gset4))
# 
# # step 2: create seurat object
# library(Seurat)
# reticulate::use_condaenv("PASCode", required = TRUE)
# library(anndata)
# ad <- anndata::read_h5ad('/home/che82/athan/PASCode/code/github_repo/data/PsychAD/adata_AD_sel_NPS_with_gxp.h5ad') # NOTE
# ad[['phenotype']] <- ad$obs['c02_PAC+']
# gene_list <- ad$var_names
# sobj <- CreateSeuratObject(
#   counts=t(ad$X),
#   assay = "X",
# )
# 
# sobj[['phenotypes']] <- 
# 
# # step 3: plot
# DotPlot(object = pbmc_small, features = cd_genes, split.by = 'phenotypes', cols="RdBu", dot.scale=10)
# 
# 
# # https://github.com/dpcook/code_snippets/blob/main/20210510_recreate_seurat_dotplot/recreate_seurat_dotplot.md
# # dot_plot <- ggplot(meta_summary, aes(x=Gene, y=seurat_clusters)) +
# #   geom_point(aes(size = Pct, fill = Avg), color="black", shape=21) +
# #   scale_size("% detected", range = c(0,6)) +
# #   scale_fill_gradientn(colours = viridisLite::mako(100),
# #                        guide = guide_colorbar(ticks.colour = "black",
# #                                               frame.colour = "black"),
# #                        name = "Average\nexpression") +
# #   ylab("Cluster") + xlab("") +
# #   theme_bw() +
# #   theme(axis.text.x = element_text(size=10, angle=45, hjust=1, color="black"),
# #         axis.text.y = element_text(size=12, color="black"),
# #         axis.title = element_text(size=14))
