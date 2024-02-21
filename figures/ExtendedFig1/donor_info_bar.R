library(ComplexHeatmap)
library(dplyr)
library(circlize)
library(RColorBrewer)
# library(scales)

setwd('/home/che82/athan/PASCode/code/github_repo/figures/fig6/')

# load data
## read celltypes
celltypes_order = read.csv("/home/che82/athan/PASCode/code/github_repo/figures/fig6/celltypes_order.csv", row.names = 1) 
## read PsychAD subclass color palette
subclass_color_palette <- read.csv("/home/che82/athan/PASCode/code/github_repo/figures/subclass_palette.csv", header=FALSE)
subclass_color_palette <- setNames(subclass_color_palette$V2[2:dim(subclass_color_palette)[1]], subclass_color_palette$V1[2:dim(subclass_color_palette)[1]])
class_color_palette <- read.csv("/home/che82/athan/PASCode/code/github_repo/figures/class_palette.csv", header=FALSE)
class_color_palette <- setNames(class_color_palette$V2[2:dim(class_color_palette)[1]], class_color_palette$V1[2:dim(class_color_palette)[1]])

################################################################################
# donor info bars
################################################################################
# dinfo = read.csv("/home/che82/athan/PASCode/code/github_repo/figures/fig6/dinfo_for_fig6.csv", row.names = 1)
dinfo = read.csv("/home/che82/athan/PASCode/code/github_repo/figures/ext_fig1/dinfo_for_extended_fig1.csv", row.names = 1)
df = read.csv("/home/che82/athan/PASCode/code/github_repo/figures/fig6/donor_PACnumber_by_celltype.csv", row.names = 1)
dinfo$c02x[dinfo$c02x=='0'] <- 'Control'
dinfo$c90x[dinfo$c90x=='0'] <- 'Control'
dinfo$c91x[dinfo$c91x=='0'] <- 'Control'
dinfo$c92x[dinfo$c92x=='0'] <- 'Control'
mat = as.matrix(df)

color_age = colorRamp2(c(min(dinfo$Age), max(dinfo$Age)), 
                       # c("#fcf5c3", "#b0dc94", "#77bdaf", "#728ab1", "#a197b4", "#2d2d2d")
                       c('#b0eba2', 'blue')
)

Na_color <- '#dadada'

column_ha = HeatmapAnnotation(
  AD = dinfo$c02x, 
  # r01x = dinfo$r01x,
  Braak_stage = dinfo$r01x,
  AD_strict_and_Resilient = dinfo$c28x,
  Sleep_WeightGain_Guilt_Suicide = dinfo$c90x, 
  WeightLoss_PMA = dinfo$c91x, 
  Depression_Mood = dinfo$c92x,
  Sex = dinfo$Sex,
  Ethnicity = dinfo$Ethnicity,
  Age = dinfo$Age,
  col = list(AD = c("AD" = "#591496", 'Na'=Na_color, "Control" = "#1f7a0f"),
             Sleep_WeightGain_Guilt_Suicide = c("Sleep_WeightGain_Guilt_Suicide" = "#0e38c2", 'Na'=Na_color,"Control" = "#addeb3"),
             WeightLoss_PMA = c("WeightLoss_PMA" = "#0e38c2", 'Na'=Na_color,"Control" = "#addeb3"),
             Depression_Mood = c('Depression_Mood'='#0e38c2', 'Na'=Na_color,'Control'='#addeb3'),
             Sex = c('Male'='#40E0D0', 'Na'=Na_color,'Female'='#FF6B00'),
             Age = color_age,
             AD_strict_and_Resilient = c('AD_strict'='#591496', 'Na'=Na_color, Control='#1f7a0f', AD_resilient='#20b8da'),
             Braak_stage = c(
               '0.0'= "#1f7a0f",
               '1.0'= "#389e26",
               '2.0'= "#6dc95d",
               '3.0'= "#7389d1",
               '4.0'= "#4969d1",
               '5.0'= "#9e61d4",
               '6.0'= "#591496",
               'Na'=Na_color),
             Ethnicity = c(
               'EUR' = '#D81B60',
               'AMR' = '#1E88E5',
               'AS' = '#004D40',
               'AFR' = '#57A860')
  )
)

# 
custom_mat <- matrix(runif(nrow(dinfo) * 10), ncol=nrow(dinfo))
pdf("/home/che82/athan/PASCode/code/github_repo/figures/fig1/donor_info_bar.pdf", pointsize=12, width = 17, height=12)
Heatmap(custom_mat, name = "mat",
        top_annotation = column_ha,
        cluster_rows=F,
        cluster_columns = F
)
dev.off()
