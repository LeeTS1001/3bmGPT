args <- commandArgs(trailingOnly = TRUE)

library(Seurat)
library(ggplot2)
library(dplyr)
library(ggradar)
library(rcdk)
library(fingerprint)
library(igraph)
library(pals)

#####Reading pre-trained data#####
pre_norm <- read.csv(file=args[1], row.names = 1, header=FALSE)
pre_meta <- read.csv(file=args[2], row.names = 1)
pre_meta$cluster <- as.character(pre_meta$cluster)
pre_embedding_min <-  -10.31254
pre_embedding_max <- 10.75763 - pre_embedding_min

#####Reading input data and Scaling#####
input_dat <- readLines(args[3])
input_dat <- unlist(strsplit(input_dat, split="\t"))
input_dat <- input_dat[-1]
v_no <- paste0("V", c(1:length(input_dat)))
input_dat <- as.numeric(input_dat)
input_dat <- input_dat - pre_embedding_min
input_dat <- round((input_dat/pre_embedding_max)*10000)
input_dat <- input_dat[match(rownames(pre_norm), v_no)]
input_smile <- readLines(args[4])
input_smile <- unlist(strsplit(input_smile, split = "\t"))[1]

#####Performing UMAP analysis#####
input_mat <- cbind(pre_norm, input_dat)
input_meta <- rbind(pre_meta, c("New", "New", "New", "None", "New", "New", input_smile))
obj <- CreateSeuratObject(counts=input_mat)
obj <- NormalizeData(obj, normalization.method = "RC")
nf <- nrow(input_mat)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures=nf)
all_f <- rownames(obj)
obj <- ScaleData(obj, features=all_f)
obj <- RunPCA(obj, features=all_f, npcs = 200)
obj <- RunUMAP(obj, reduction="pca", dims=1:100)
obj@meta.data <- cbind(obj@meta.data, input_meta)

#####Generating UMAP figure with functional category#####
gg_dat <- data.frame(obj@reductions$umap@cell.embeddings[,1], obj@reductions$umap@cell.embeddings[,2], obj@meta.data)
colnames(gg_dat)[1:2]<- c("UMAP_1", "UMAP_2")
gg_dat$point <- "Standard"
gg_dat$point[10001] <- "New"

cen_UMAP_1 <- as.numeric(tapply(gg_dat$UMAP_1[-10001], gg_dat$group_cluster[-10001], mean))
cen_UMAP_2 <- as.numeric(tapply(gg_dat$UMAP_2[-10001], gg_dat$group_cluster[-10001], mean))
cen_names <- c("Catalysis", "Nuclear Receptor", "Phospholipase", "Protease", "Protein Kinase", "Transferase")
New_ann <- as.numeric(gg_dat[gg_dat$point=="New",c("UMAP_1", "UMAP_2")])

first_dir <- paste0(args[5], "/", args[6], "_UMAP_with_Biological_Function.tif")
tiff(filename = first_dir, width = 14, height = 10,units = "cm", res = 400)
ggplot(gg_dat[-10001,], aes(x=UMAP_1, y=UMAP_2)) + geom_point(color="grey", size=0.3) + 
  stat_ellipse(data=gg_dat[-10001,], aes(color=group_cluster, fill=group_cluster), geom="polygon", alpha=0.3, type="t") + 
  scale_fill_manual(values=c("#e16766", "#f7b16b", "#98c37c", "#fdd962", "#74a4cf", "#8e7dc4")) +
  scale_color_manual(values=c("#e16766", "#f7b16b", "#98c37c", "#fdd962", "#74a4cf", "#8e7dc4")) +
  theme_bw() + 
  geom_point(data=gg_dat[gg_dat$point=="New",], color="red", shape=18, size=5) +
  theme(axis.ticks = element_blank(), axis.text = element_blank(),  text=element_text(size=20), panel.grid = element_blank()) + NoLegend()
dev.off()

#####Generating radar chart by distance between input Data and functional groups#####
cen_dat <- data.frame(cen_UMAP_1, cen_UMAP_2, cen_names)
colnames(cen_dat) <- c("UMAP_1", "UMAP_2", "Type")
cen_dat$dis <- sqrt((cen_dat$UMAP_1 - gg_dat[10001,"UMAP_1"])^2 + (cen_dat$UMAP_2 - gg_dat[10001,"UMAP_2"])^2)
cen_dat$dis <- 1/cen_dat$dis
cen_dat$dis <- cen_dat$dis/max(cen_dat$dis)
gg_dat <- c(1, cen_dat$dis)
names(gg_dat) <- c("group", cen_dat$Type)
gg_dat <- data.frame(gg_dat)
gg_dat <- data.frame(t(gg_dat))
colnames(gg_dat)[grep("Nuclear", colnames(gg_dat))] <- "Nuclear\nReceptor"
colnames(gg_dat)[grep("Kinase", colnames(gg_dat))] <- "Protein\nKinase"

second_dir <- paste0(args[5], "/", args[6], "_Radar_Chart_For_Biological_Function.tif")
tiff(filename = second_dir, width = 14, height = 10,units = "cm", res = 400)
ggradar(gg_dat, grid.min=0, grid.mid=0.5, grid.max=1, group.colours = "#e16766", values.radar=c("0", "0.5", "1"), 
        grid.label.size = 0,
        axis.label.size = 5) + 
  theme_void() +
  theme(axis.ticks = element_blank(), axis.text = element_blank(), text=element_text(size=10)) + NoLegend()
dev.off()

#####Calculating ligand Similarity by Tanimoto Coefficient#####
bg_smile <- pre_meta$smiles
smiles_no <- c(1:length(bg_smile))
smiles_no <- smiles_no[nchar(bg_smile)!=0]
bg_smile <- bg_smile[nchar(bg_smile)!=0]
mA <- parse.smiles(input_smile)[[1]]
molA <- get.fingerprint(mA, type="extended")
dis_dat <- rep(0, length(bg_smile))
for(i in 1:length(bg_smile)){
  mB <- parse.smiles(bg_smile[i])[[1]]
  molB <- get.fingerprint(mB, type="extended")
  dis_dat[i] <- distance(molA, molB, method="tanimoto")
}

#####Calculating average ligand similarity for binning#####
umap_dat <- data.frame(obj@reductions$umap@cell.embeddings[,1], obj@reductions$umap@cell.embeddings[,2])
umap_dat <- umap_dat[c(smiles_no, 10001),]
umap_dat$dis <- c(dis_dat,1)
colnames(umap_dat) <- c("UMAP1", "UMAP2", "Sim")

d1 <- (max(umap_dat$UMAP1) - min(umap_dat$UMAP1))/100
d2 <- (max(umap_dat$UMAP2) - min(umap_dat$UMAP2))/100
qt_n1 <- seq(min(umap_dat$UMAP1), max(umap_dat$UMAP1), d1)
qt_n2 <- seq(min(umap_dat$UMAP2), max(umap_dat$UMAP2), d2)

ind <- c(99:1)
umap_dat$group1 <- 100
umap_dat$group2 <- 100

for(i in ind){
  umap_dat$group1[umap_dat$UMAP1 < qt_n1[i]] <- i
  umap_dat$group2[umap_dat$UMAP2 < qt_n2[i]] <- i
}

#####Generating binned UMAP figure for ligand similarity#####
input_n <- nrow(umap_dat)
umap_dat$bin <- paste0(umap_dat$group1, "_", umap_dat$group2)
gg_dat <- data.frame(as.numeric(tapply(umap_dat$Sim[-input_n], umap_dat$bin[-input_n], mean)), 
                     names(tapply(umap_dat$Sim[-input_n], umap_dat$bin[-input_n], mean)))
colnames(gg_dat) <- c("sim", "bin")
bin_n <- as.numeric(unlist(strsplit(gg_dat$bin, split = "_")))
gg_dat$UMAP1 <- bin_n[seq(1,length(bin_n), 2)]
gg_dat$UMAP2 <- bin_n[seq(2,length(bin_n), 2)]
gg_dat$Input <- "Other"
gg_dat$Input[gg_dat$bin == umap_dat[nrow(umap_dat), "bin"]] <- "Input"
colnames(gg_dat)[1] <- "Avg.TC"
gg_dat$Avg.TC[gg_dat$Avg.TC>0.5] <- 0.5

third_dir <- paste0(args[5], "/", args[6], "_UMAP_by_Ligand_Similarity.tif")
tiff(filename = third_dir, width = 14, height = 10,units = "cm", res = 400)
ggplot(gg_dat, aes(x=UMAP1, y=UMAP2, fill=Avg.TC)) + geom_point(shape=22, size=0.9, stroke=0.3) + theme_bw() +
  scale_fill_gradientn(colours = pals::jet(10)) +
  geom_point(data=gg_dat[gg_dat$Input=="Input",], color="red", size=5, shape=1, stroke=3) + theme_void() +
  theme(legend.position = "bottom")
dev.off()

#####Saving top20 close interaction data#####
umap_dat <- data.frame(obj@reductions$umap@cell.embeddings[,1], obj@reductions$umap@cell.embeddings[,2], obj@meta.data)
colnames(umap_dat)[1:2]<- c("UMAP_1", "UMAP_2")
umap_dat$dis <- sqrt((umap_dat$UMAP_1 - umap_dat[10001,"UMAP_1"])^2 + (umap_dat$UMAP_2 - umap_dat[10001,"UMAP_2"])^2)
umap_dat$ind <- paste0(umap_dat$gene, "_", umap_dat$species, "_",  umap_dat$ligand_id)
umap_dat <- umap_dat[order(umap_dat$dis),]
umap_dat <- umap_dat[!(duplicated(umap_dat$ind)),]
save_dat <- umap_dat[c(1:21),]
rownames(save_dat) <- c("Input", paste0("near_data_", c(1:20)))
save_dat <- save_dat[,-c(3,4,5,9, 14)]
save_dat$cluster[1] <- "Input"
save_dat$group_cluster[1] <- "Input"
save_dat$ligand_id[1] <- "Input"
save_dat$gene[1] <- "Input"
save_dat$species[1] <- "Input"
colnames(save_dat)[c(3,4,5,6,7,8,9)] <- c("Cluster_No", "Target", "Species", "Functional_Group", "Ligand_ID", "SMILES", "Distance")

forth_dir <- paste0(args[5], "/", args[6], "_Top20_Close_Data.txt")
write.table(save_dat, file=forth_dir, quote = FALSE, sep = "\t")

#####Generating network graph for ligands from top20 close interactions#####
umap_dat <- umap_dat[-1,]
umap_dat <- umap_dat[1:20,]
umap_dat <- umap_dat[!duplicated(umap_dat$ligand_id),]
use_smile <- c(input_smile, umap_dat$smiles)

mol <- list()
molF <- list()
for(i in 1:length(use_smile)){
  mol[[i]] <- parse.smiles(use_smile[i])[[1]]
  molF[[i]] <- get.fingerprint(mol[[i]], type="extended")
}

sim_dat <- matrix(0, nrow=length(use_smile), ncol=length(use_smile))
for(i in 1:length(use_smile)){
  for(j in 1:length(use_smile)){
    sim_dat[i,j] <- distance(molF[[j]], molF[[i]], method="tanimoto")
  }
}

for(i in 1:nrow(sim_dat)){
  sim_dat[i,i] <- 0
  sim_dat[i, (1:i-1)] <- 0
}

cutoff <- as.numeric(sim_dat)[order(as.numeric(sim_dat), decreasing=TRUE)][10]
if(cutoff>0.3){cutoff <- cutoff} else{cufoff <- 0.3}

sim_dat <- sim_dat>=cutoff
rownames(sim_dat) <- as.character(c("Input", umap_dat$ligand_id))
colnames(sim_dat) <- as.character(c("Input", umap_dat$lig))

sim_result <- apply(sim_dat, 1, function(x) which(x))
link_list <- names(unlist(sim_result))
link_list <- unlist(strsplit(link_list, split = "[.]"))
sole_node <- rownames(sim_dat)[!(rownames(sim_dat) %in% link_list)]

g1 <- graph(edges=link_list, directed=F , isolates = sole_node) 

grades <- names(V(g1))
grades <- factor(grades=="Input")
levels(grades) <- c("#fdd962", "#e16766")
use_color <- as.character(grades)
V(g1)$label.cex <- 0.8

fifth_dir <- paste0(args[5], "/", args[6], "Network_of_Top20_Close_Ligands.tif")
tiff(filename = fifth_dir, width = 10, height = 10,units = "cm", res = 400)
par(bg="white", mar=c(0,0,0,0))
set.seed(12)
plot(g1, vertex.size=20, vertex.label.color="black", vertex.color=use_color, edge.width=1, edge.color="black")
dev.off()

