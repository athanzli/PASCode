library(pheatmap)
setwd('/home/che82/athan/ProjectPASCode/figures/fig2')

color_palette <- colorRampPalette(c("#faf7c8", "orange", "red"))(n = 100)

pdf("/home/che82/athan/PASCode/code/github_repo/figures/s2/SEAAD_celltype_label_transfer.pdf", pointsize=12, width = 15, height=15)
perc = read.table('perc.s8.SEAAD.csv',header=T,row.names=1,sep=',')
pheatmap(perc,
         main='SEA-AD cell type label transferring', 
         color = color_palette,
         fontsize = 24
)
dev.off()

pdf("/home/che82/athan/PASCode/code/github_repo/figures/s2/ROSMAP_celltype_label_transfer.pdf", pointsize=12, width = 15, height=15)
perc = read.table('perc.s8.ROSMAP.csv',header=T,row.names=1,sep=',')
pheatmap(perc,
         main='ROSMAP cell type label transferring', 
         color = color_palette,
         fontsize = 24
)
dev.off()
