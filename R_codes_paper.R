# Packages for data manipulation and analysis
library(knitr)
library(xtable)
library(ISLR)
library(skimr)
library(tidyverse)
library(flextable)

# Packages for data visualization
library(ggplot2)
library(ggpubr)
library(gridExtra)

# Packages for model evaluation
library(caret)
library(yardstick)
library(flashlight)
library(iml)
library(rpart)
library(rpart.plot)
library(xgboost)

# Packages for kohonen/SOM analysis
library(class)
library(kohonen)
library(withr)
library(shapviz)
library(kernelshap)
library(randomForest)
library(reshape2)
library(gridExtra)
library(factoextra)
library(aweSOM)
library(RColorBrewer)
library(viridis)
library(grid)


# Data Partitioning and Model Configuration
load("Sample.RData")
set.seed(423)
partition <- createDataPartition(y = datos$Severity, p = 0.7, list = FALSE)
datos_train <- datos[partition, ]
datos_test <- datos[-partition, ]
prop.table(table(datos_train$Severity))
prop.table(table(datos_test$Severity))

## Original
tcontrol_original = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                                 savePredictions = "final",
                                 returnResamp =  "final",
                                 classProbs = TRUE,  
                                 seed = set.seed(423),
                                 summaryFunction = twoClassSummary)
## Undersampling
tcontrol_under = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              savePredictions = "final", 
                              returnResamp =  "final",
                              classProbs = TRUE,  
                              seed = set.seed(423),
                              summaryFunction = twoClassSummary,
                              sampling = "down")
## Oversampling
tcontrol_over = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                             savePredictions = "final", 
                             returnResamp =  "final",
                             classProbs = TRUE,  
                             seed = set.seed(423),
                             summaryFunction = twoClassSummary, 
                             sampling = "up")
## SMOTE
require(themis)
tcontrol_smote = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                              savePredictions = "final",
                              returnResamp =  "final",
                              classProbs = TRUE,  
                              seed = set.seed(423),
                              summaryFunction = twoClassSummary, 
                              sampling = "smote")
## ROSE
require(ROSE)
tcontrol_ROSE = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                             savePredictions = "final", 
                             returnResamp =  "final",
                             classProbs = TRUE, 
                             seed = set.seed(423),
                             summaryFunction = twoClassSummary, 
                             sampling = "rose")
## Cost-sensitive weights
model_weights <- ifelse(datos_train$Severity == "NC",
                        (1/table(datos_train$Severity)[1]) * 0.5,
                        (1/table(datos_train$Severity)[2]) * 0.5)

## Logistic

mdl_log_original <- caret::train(
   Severity ~ ., data = datos_train, method = "glm",
   family = 'binomial', trControl = tcontrol_original)

mdl_log_weighted <- caret::train(
   Severity ~ ., data = datos_train, method = "glm",
   family = 'binomial', weights = model_weights,
   trControl = tcontrol_original)

mdl_log_under <- caret::train(
   Severity ~ ., data = datos_train, method = "glm",
   family = 'binomial', trControl = tcontrol_under)

mdl_log_over <- caret::train(
   Severity ~ ., data = datos_train, method = "glm",
   family = 'binomial', trControl = tcontrol_over)

mdl_log_smote <- caret::train(
   Severity ~ ., data = datos_train, method = "glm",
   family = 'binomial', trControl = tcontrol_smote)

## Random Forest

mdl_rf_original <- caret::train(
   Severity ~ ., data = datos_train, method = "rf",
   metric = "ROC", trControl = tcontrol_original)

mdl_rf_weighted <- caret::train(
   Severity ~ ., data = datos_train, method = "rf",
   metric = "ROC", weights = model_weights,
   trControl = tcontrol_original)

mdl_rf_under <- caret::train(
   Severity ~ ., data = datos_train, method = "rf",
   metric = "ROC", trControl = tcontrol_under)

mdl_rf_over <- caret::train(
   Severity ~ ., data = datos_train, method = "rf",
   metric = "ROC", trControl = tcontrol_over)

mdl_rf_smote <- caret::train(
   Severity ~ ., data = datos_train, method = "rf",
   metric = "ROC", trControl = tcontrol_smote)

## XGBoost

mdl_xgb_original <- caret::train(
   Severity ~ ., data = datos_train, method = "xgbTree",
   metric = "ROC", trControl = tcontrol_original, 
   verbosity = 0)

mdl_xgb_weighted <- caret::train(
   Severity ~ ., data = datos_train, method = "xgbTree",
   metric = "ROC", weights = model_weights,
   trControl = tcontrol_original, 
   verbosity = 0)

mdl_xgb_under <- caret::train(
   Severity ~ ., data = datos_train, method = "xgbTree",
   metric = "ROC", trControl = tcontrol_under, 
   verbosity = 0)

mdl_xgb_over <- caret::train(
   Severity ~ ., data = datos_train, method = "xgbTree",
   metric = "ROC", trControl = tcontrol_over, 
   verbosity = 0)

mdl_xgb_smote <- caret::train(
   Severity ~ ., data = datos_train, method = "xgbTree",
   metric = "ROC", trControl = tcontrol_smote, 
   verbosity = 0)

# Comparison of class balancing methods for models

resamp <- resamples(list("log-original" = mdl_log_original,
                            "log-under" = mdl_log_under,
                            "log-over" = mdl_log_over,
                            "log-smote" = mdl_log_smote,
                            "rf-original" = mdl_rf_original,
                            "rf-under" = mdl_rf_under,
                            "rf-over" = mdl_rf_over,
                            "rf-smote" = mdl_rf_smote,
                            "xgb-original" = mdl_xgb_original,
                            "xgb-under" = mdl_xgb_under,
                            "xgb-over" = mdl_xgb_over,
                            "xgb-smote" = mdl_xgb_smote))
summary(resamp)
resamp$models <- as.factor(resamp$models)
bwplot(resamp, col = resamp$models, groups = resamp$models, metric=c("Sens","Spec","ROC"))


# RF+undersampling results

mdl_rf_under
preds_rfu <- bind_cols(
   predict(mdl_rf_under, newdata = datos_test, type = "prob"),
   Predicted = predict(mdl_rf_under, newdata = datos_test, type = "raw"),
   Actual = datos_test$Severity)
cm_rfu <- caret::confusionMatrix(preds_rfu$Predicted, 
                            reference = preds_rfu$Actual,positive="FS")
cm_rfu

preds_rfu$CM<-""
preds_rfu[preds_rfu$Predicted=="FS"&preds_rfu$Actual=="FS",5]<-"TP"
preds_rfu[preds_rfu$Predicted=="NS"&preds_rfu$Actual=="NS",5]<-"TN"
preds_rfu[preds_rfu$Predicted=="NS"&preds_rfu$Actual=="FS",5]<-"FN"
preds_rfu[preds_rfu$Predicted=="FS"&preds_rfu$Actual=="NS",5]<-"FP"
table(preds_rfu$CM)
preds_rfu$CM<-factor(preds_rfu$CM)
table(datos_test$Severity)


# Settings for using iml

predictor_rf_under <- Predictor$new(model = mdl_rf_under, data = datos_test, 
                                    y = "Severity", type="prob")

predictor_rf_under2 <- Predictor$new(model = mdl_rf_under, data = datos_test, 
                                     y = "Severity", type="prob", class = "FS")

# Global interpretation with iml

imp_rf_under <- FeatureImp$new(predictor_rf_under, loss = "ce")
plot(imp_rf_under)

effs_rf_u <- FeatureEffects$new(predictor_rf_under2)
plot(effs_rf_u)

interact_all_rfu_iml <- Interaction$new(predictor_rf_under2)
plot(interact_all_rfu_iml)


# Local interpretation: Shapley values

## single observation
set.seed(123)
shapley.1<-Shapley$new(predictor_rf_under2, x.interest = datos_test[1,-17]) 
plot(shapley.1)
shapley.1$results$phi

## all observations
VS_rfu<-as.data.frame(matrix(0,nrow(datos_test),ncol(datos_test)-1),col.names=colnames(datos_test[,-17]))
it<-1
for(it in 1:nrow(datos_test)){
  set.seed(123)
  VS_rfu[it,]<-Shapley$new(predictor_rf_under2, x.interest = datos_test[it,-17])$results$phi
}
dim(VS_rfu) 
datos_test_rfu_vs<-cbind(datos_test,preds_rfu,VS_rfu)

# SOM maps

aux_rf_under_S <- datos_test_rfu_vs[,23:38] #Shapley values
aux_rf_under_X <- datos_test_rfu_vs[,1:16] #Covariables
aux_rf_under_real <- datos_test_rfu_vs[,21] #Real output
aux_rf_under_p <- datos_test_rfu_vs[,20] #Predicted output

## How to determine the optimal grid and radius?

u_matrix_score <- function(som_model) {
  codes <- som_model$codes[[1]]
  grid_pts <- som_model$grid$pts
  n_nodes <- nrow(codes)
  # Calculate distances between neighboring neurons
  neighbor_dists <- c()
  for (i in 1:n_nodes) {
    for (j in i:n_nodes) {
      # Check if units i and j are neighbors (Manhattan distance == 1)
      if (sum(abs(grid_pts[i, ] - grid_pts[j, ])) == 1) {
        dist_ij <- sqrt(sum((codes[i, ] - codes[j, ])^2))
        neighbor_dists <- c(neighbor_dists, dist_ij)
      }
    }
  }
  # Return standard deviation of neighbor distances (higher = better cluster contrast)
  sd(neighbor_dists)
}

evaluate_som_grids_awe<- function(data, grid_sizes = c(4:6), topo = "hexagonal", rlen = 200) {
  results_awe <- data.frame(GridSize = character(), QE = numeric(), TE = numeric(), 
                            KLE = numeric(), UMatrixSD = numeric(), stringsAsFactors = FALSE)
  for (x in grid_sizes) {
    for (y in grid_sizes) {
      cat("\nTraining SOM grid", x, "x", y, "...\n")
      grid <- somgrid(xdim = x, ydim = y, topo = topo)
      set.seed(123)
      som_model <- som(data, grid = grid, rlen = rlen, radius = max(x/2, y/2), keep.data = TRUE,
                       maxNA.fraction = 0.2)
      # Compute metrics
      SQ <- somQuality(som_model, data)
      qe <- SQ$err.quant
      te <- SQ$err.topo
      kle <- SQ$err.kaski
      u_sd <- u_matrix_score(som_model)  # U-Matrix SD
      # Save to results
      results_awe <- rbind(results_awe, data.frame(GridSize = paste0(x, "x", y), 
                                                   QE = qe, TE = te, KLE = kle, 
                                                   UMatrixSD = u_sd))
      cat("QE:", qe, "| TE:", te, "| KLE:", kle, "| UMatrixSD:", u_sd, "\n")
    }
  }
  return(results_awe)
}

dat<-scale(as.matrix(aux_rf_under_S))
dat2<-dat[complete.cases(dat),]
results_awe <- evaluate_som_grids_awe(dat, grid_sizes = 4:7)
scale_u <- function(x) {
  (x - min(x)) / (max(x) - min(x))}
results_awe$CombinedScore <- -scale_u(results_awe$KLE) + scale_u(results_awe$UMatrixSD)
results_awe_ord <- results_awe[order(-results_awe$CombinedScore), ]
print(results_awe_ord) #7x4 is optimal

evaluate_som_radius <- function(data, grid = somgrid(7, 4, topo = "hexagonal"), rlen = 200, rad_sizes=seq(2,4,0.25)) {
results_rad <- data.frame(RadSize = character(), QE = numeric(), TE = numeric(), KLE = numeric(), EVP=numeric(), UMSD=numeric(), stringsAsFactors = FALSE)
  for (i in rad_sizes) {
      set.seed(123)
      som_model <- som(data, grid = grid, rlen = rlen, radius = c(i,-i), keep.data = TRUE,
                       maxNA.fraction = 0.2)
      # Compute metrics
      SQ<-somQuality(som_model, data)
      qe <- SQ$err.quant
      te <- SQ$err.topo
      kle<- SQ$err.kaski
      evp<- SQ$err.varratio
      umsd <- u_matrix_score(som_model)
      # Save to results
      results_rad <- rbind(results_rad, data.frame(GridSize = paste0("radius=",i), QE = qe, TE = te, KLE=kle, EVP=evp, UMSD=umsd))
      cat("QE:", qe, "| TE:", te, "| KLE:", kle,"| EVP:", evp,"| UMSD:", umsd,  "\n")
    }
  return(results_rad)
}

results_rad <- evaluate_som_radius(dat)
print(results_rad) #3.25 is optimal

## Training SOM

xdim<-7
ydim<-4
rad<-3.25 #or rad<-max(xdim/2,ydim/2)
set.seed(1003)
som_grid <- kohonen::somgrid(xdim = xdim, ydim = ydim, topo="hexagonal")
som_model <- kohonen::supersom(scale(as.matrix(aux_rf_under_S)), grid=som_grid, rlen=200, alpha=c(0.05,0.01), radius = c(rad,-rad), keep.data = TRUE, maxNA.fraction = 0.2)
plot(som_model, type = "changes")
plot(som_model, type = "mapping", pchs = 19, shape = "round")
plot(som_model, type = "counts" )
plot(som_model, type = "dist.neighbours")

## Clustering neurons

nk<-19 

som_cluster_km <- kmeans(som_model$codes[[1]], nk) 
superclasses_km <- som_cluster_km$cluster

som_cluster_pam <- cluster::pam(som_model$codes[[1]], nk)
superclasses_pam <- som_cluster_pam$clustering

som_cluster_h <- hclust(dist(som_model$codes[[1]]), "complete")
superclasses_h <- cutree(som_cluster_h, nk)

fviz_nbclust(som_model$codes[[1]], kmeans, method = "wss", k.max = nrow(som_model$codes[[1]])-1)
aweSOMscreeplot(som = som_model, method = "pam", nclass = nk)
aweSOMscreeplot(som = som_model, method = "hierarchical", nclass = nk)

aweSOMsilhouette(som_model, superclasses_km)
aweSOMsilhouette(som_model, superclasses_pam)
aweSOMsilhouette(som_model, superclasses_h)

aweSOMdendrogram(clust = som_cluster_h, nclass = nk)
plot(som_cluster_h,cex = 0.6, hang = -1)
rect.hclust(som_cluster_h, k=nk, border="red") 

nk<-10 #optimal number of clusters
palette <- c('#e377c2', "#98C80C",  "#33A5BF", "#F7D940",'#9467bd','#ff7f0e', '#2ca02c', '#d62728',  '#8c564b',"#F25F73" ,"#888E94", '#76b7b2',"#1f77b4")
palette2 <- c( "#F25F73","#33A5BF")
aweSOMdendrogram(clust = som_cluster_h, nclass = nk)
plot(som_cluster_h,cex = 0.6, hang = -1)
rect.hclust(som_cluster_h, k=nk, border="red") 

## SOM with k-means
set.seed(123) 
som_cluster_km <- kmeans(som_model$codes[[1]], nk) 
superclasses_km <- som_cluster_km$cluster
table(superclasses_km)
table(factor(superclasses_km[som_model$unit.classif]))
plot(som_model, type = "codes", bgcol = palette[superclasses_km], main = "Cluster Map K-means", shape = "straight") 
add.cluster.boundaries(som_model, superclasses_km)

## SOM with k-medians
set.seed(123) 
som_cluster_pam <- cluster::pam(som_model$codes[[1]], nk)
superclasses_pam <- som_cluster_pam$clustering
table(superclasses_pam)
table(factor(superclasses_pam[som_model$unit.classif]))
plot(som_model, type = "codes", bgcol = palette[superclasses_pam], main = "Cluster Map K-medians", shape = "straight") 
add.cluster.boundaries(som_model, superclasses_pam)

## SOM with hclust
set.seed(123) 
som_cluster_h <- hclust(dist(som_model$codes[[1]]), "complete")
superclasses_h <- cutree(som_cluster_h, nk)
table(superclasses_h)
table(factor(superclasses_h[som_model$unit.classif]))
plot(som_model, type = "codes", bgcol = palette[superclasses_h], main = "Cluster Map Hierarchical", shape = "straight") 
add.cluster.boundaries(som_model, superclasses_h)

## new dataframes with clustering information
shap_clusters_km_rf_under = data.frame(datos_test_rfu_vs[,c(1:17,19:38)], cell=som_model$unit.classif, cluster = factor(superclasses_km[som_model$unit.classif]))
shap_clusters_pam_rf_under = data.frame(datos_test_rfu_vs[,c(1:17,19:38)], cell=som_model$unit.classif, cluster = factor(superclasses_pam[som_model$unit.classif]))
shap_clusters_h_rf_under = data.frame(datos_test_rfu_vs[,c(1:17,19:38)], cell=som_model$unit.classif, cluster = factor(superclasses_h[som_model$unit.classif]))

## Comparation cluster methods
tab_km<-table(factor(superclasses_km[som_model$unit.classif]))
tab_pam<-table(factor(superclasses_pam[som_model$unit.classif]))
tab_h<-table(factor(superclasses_h[som_model$unit.classif]))
barplot(sort(tab_km,decreasing=TRUE))
barplot(sort(tab_pam,decreasing=TRUE))
barplot(sort(tab_h,decreasing=TRUE))
aweSOMplot(som = som_model, type = "Color", data = shap_clusters_km_rf_under, 
           variables = "FS", superclass = superclasses_km)
aweSOMplot(som = som_model, type = "Color", data = shap_clusters_pam_rf_under, 
           variables = "FS", superclass = superclasses_pam)
aweSOMplot(som = som_model, type = "Color", data = shap_clusters_h_rf_under, 
           variables = "FS", superclass = superclasses_h)
aweSOMplot(som = som_model, type = "CatBarplot", data = shap_clusters_km_rf_under, 
           variables = "Predicted", superclass = superclasses_km, palsc="rainbow", showSC = TRUE)
aweSOMplot(som = som_model, type = "CatBarplot", data = shap_clusters_pam_rf_under, 
           variables = "Predicted", superclass = superclasses_pam, palsc="rainbow", showSC = TRUE)
aweSOMplot(som = som_model, type = "CatBarplot", data = shap_clusters_h_rf_under, 
           variables = "Predicted", superclass = superclasses_h, palsc="rainbow", showSC = TRUE)

## Summary for cluters (from hclust)

for(i in 1:10){
  cat("\nSummary for cluster ", i , "\n")
  print(summary(shap_clusters_h_rf_under[shap_clusters_h_rf_under$cluster==i,1:16]))
  cat("\nConfusion matrix for cluster ", i , "\n")
  print(caret::confusionMatrix(shap_clusters_h_rf_under[shap_clusters_h_rf_under$cluster==i,"Predicted"], shap_clusters_h_rf_under[shap_clusters_h_rf_under$cluster==i,"Actual"],positive="FS"))
  cat("\nAverage of prob(FS) for cluster ", i , "\n")
  print(round(mean(shap_clusters_h_rf_under[shap_clusters_h_rf_under$cluster==i,18]),4))
  cat("\nMean of Shapley values for cluster ", i , "\n")
  print(sort(round(colMeans(shap_clusters_h_rf_under[shap_clusters_h_rf_under$cluster==i,22:37]),4)))
  cat("\n_________________________________________\n")
}

## Decision tree for interpreting clusters

m_tree <- rpart::rpart(cluster ~ .,
                       data = shap_clusters_h_rf_under[, c(1:16, 39)],
                       maxdepth = 30,
                       cp = 0.0010,
                       parms = list(split = "gini"))  

prp_colored_by_direction <- function(tree, left_col = "forestgreen", right_col = "firebrick", ...) {
  # Get the frame of the tree
  frame <- tree$frame
  n <- nrow(frame)
  # Get the node numbers
  nodes <- as.numeric(rownames(frame))
  # Initialize branch color vector
  branch_colors <- rep(NA, n)
  # Loop through internal nodes
  for (i in seq_along(nodes)) {
    node <- nodes[i]
    if (frame$var[i] != "<leaf>") {
      left_node <- node * 2
      right_node <- node * 2 + 1
      # Find indices of left and right nodes
      left_index <- which(nodes == left_node)
      right_index <- which(nodes == right_node)
      # Assign colors
      if (length(left_index)) branch_colors[left_index] <- left_col
      if (length(right_index)) branch_colors[right_index] <- right_col
    }
  }
  # Plot with custom branch colors
  prp(tree, branch.col = branch_colors, ...)
}

prp_colored_by_direction(m_tree, type = 0, branch.type = 5, extra = 2, tweak = 1.3, branch.tweak = 0.5, split.cex = 1.1)


## SOM figures

### Grid 7x4 with clusters

par(mar = c(1, 1, 1, 1))
plot(som_model, type = "mapping", bgcol = palette[superclasses_h], main = "Cluster Map Hierarchical", shape = "straight",cex=0) 
add.cluster.boundaries(som_model, superclasses_h)
grid_coords <- som_model$grid$pts
text(grid_coords[, 1], grid_coords[, 2], labels = superclasses_h, cex = 1, col = "black")

### Grid 7x4 with means 
sev2<-tapply(as.numeric(shap_clusters_h_rf_under$Severity)-1, shap_clusters_h_rf_under$cluster, mean)
sev3<-sev2[superclasses_h]
tf<-sev3>mean(as.numeric(shap_clusters_h_rf_under$Severity)-1)
superclass2_h<-factor(ifelse(tf=="FALSE", "NS", "FS"))
par(mar = c(1, 1, 1, 1))
plot(som_model, type = "mapping", bgcol = palette2[superclass2_h], main = "", shape = "straight",cex=0) 
add.cluster.boundaries(som_model, superclasses_h)
grid_coords <- som_model$grid$pts
text(grid_coords[, 1], grid_coords[, 2], labels = round((sev3)*100,2), cex = 1, col = "black")


## Shapley values figures: TP, FP, TN, FN 

covariates<-names(shap_clusters_h_rf_under[,c(1:16)])
get_max_prop_label <- function(f, varname) {
  props <- prop.table(table(f))
  max_label <- names(which.max(props))
  max_value <- 100*max(props)
  label_expr <- bquote(bold(.(varname)) * ": " * .(format(max_value, digits=2)) * "% (" * .(max_label) * ")")
  deparse(label_expr)  
  #sprintf("%s: %.2f (%s)", varname, max_value, max_label)
}
cm<-caret::confusionMatrix(shap_clusters_h_rf_under[,"Predicted"], shap_clusters_h_rf_under[,"Actual"],positive="FS")
cm

phi.TP<-round(colMeans(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TP",c(22:37)]),6)
val.TP<-mapply(get_max_prop_label, shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TP",c(1:16)], varname = covariates)
tittle.TP<-paste0("TP: True Positive (Pred: FS; Real: FS); n=", format(nrow(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TP",]), big.mark = ","))
plotvs.TP<-ggplot(data.frame(var = covariates, phi = phi.TP, value = val.TP), 
      aes(x = reorder(value, abs(phi)),
          y=phi, 
          fill = ifelse(phi>0,'#FF0000','#0000FF'))) + 
  geom_bar(stat = "identity", alpha = 0.8) +
  # scale_fill_brewer(palette="Set1") +
  coord_flip() +
  labs(title = tittle.TP )+
  guides(fill = "none") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(name="Shapley Values",breaks = pretty(range(phi.TP), n = 5)) +
  scale_fill_manual(values=c("#FF0000" = "#FF0000", "#0000FF" = "#0000FF"))+ 
  scale_x_discrete(labels = function(x) parse(text = x)) +  
    theme(text = element_text(size = 15),
          plot.title = element_text(hjust = 0.5, size = 17),
          plot.title.position = "plot",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_blank(),
          axis.text.y = element_text(hjust = 0))
text.TP<-paste0("Probs - True Positive\n(PPV: ", round(cm$byClass["Pos Pred Value"],4), ")\n(Sens: ",round(cm$byClass["Sensitivity"],4),")")
plotpr.TP<-ggplot(data = shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TP",], 
             aes(x = FS, y = text.TP)) +
                geom_boxplot(alpha=0.3) +
                xlab("") +
                ylab("") +
                labs(title = "") +
                scale_fill_discrete(guide="none") +
                # geom_vline(xintercept = 0) +
                theme_minimal() +
                theme(text = element_text(size = 15))

phi.FP<-round(colMeans(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FP",c(22:37)]),6)
val.FP<-mapply(get_max_prop_label, shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FP",c(1:16)], varname = covariates)
tittle.FP<-paste0("FP: False Positive (Pred: FS; Real: NS); n=", format(nrow(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FP",]), big.mark = ","))
plotvs.FP<-ggplot(data.frame(var = covariates, phi = phi.FP, value = val.FP), 
      aes(x = reorder(value, abs(phi)),
          y=phi, 
          fill = ifelse(phi>0,'#FF0000','#0000FF'))) + 
  geom_bar(stat = "identity", alpha = 0.8) +
  # scale_fill_brewer(palette="Set1") +
  coord_flip() +
  labs(title = tittle.FP )+
  guides(fill = "none") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(name="Shapley Values",breaks = pretty(range(phi.FP), n = 5)) +
  scale_fill_manual(values=c("#FF0000" = "#FF0000", "#0000FF" = "#0000FF"))+ 
  scale_x_discrete(labels = function(x) parse(text = x)) +  
    theme(text = element_text(size = 15),
          plot.title = element_text(hjust = 0.5, size = 17),
          plot.title.position = "plot",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_blank(),
          axis.text.y = element_text(hjust = 0))
text.FP<-paste0("Probs - False Positive\n(PPV: ", round(cm$byClass["Pos Pred Value"],4), ")\n(Spec: ",round(cm$byClass["Specificity"],4),")")
plotpr.FP<-ggplot(data = shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FP",], 
             aes(x = FS, y = text.FP)) +
                geom_boxplot(alpha=0.3) +
                xlab("") +
                ylab("") +
                labs(title = "") +
                scale_fill_discrete(guide="none") +
                # geom_vline(xintercept = 0) +
                theme_minimal() +
                theme(text = element_text(size = 15))

phi.TN<-round(colMeans(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TN",c(22:37)]),6)
val.TN<-mapply(get_max_prop_label, shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TN",c(1:16)], varname = covariates)
tittle.TN<-paste0("TN: True Negative (Pred: NS; Real: NS); n=", format(nrow(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TN",]), big.mark = ","))
plotvs.TN<-ggplot(data.frame(var = covariates, phi = phi.TN, value = val.TN), 
      aes(x = reorder(value, abs(phi)),
          y=phi, 
          fill = ifelse(phi>0,'#FF0000','#0000FF'))) + 
  geom_bar(stat = "identity", alpha = 0.8) +
  # scale_fill_brewer(palette="Set1") +
  coord_flip() +
  labs(title = tittle.TN )+
  guides(fill = "none") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(name="Shapley Values",breaks = pretty(range(phi.TN), n = 5)) +
  scale_fill_manual(values=c("#FF0000" = "#FF0000", "#0000FF" = "#0000FF"))+ 
  scale_x_discrete(labels = function(x) parse(text = x)) +  
    theme(text = element_text(size = 15),
          plot.title = element_text(hjust = 0.5, size = 17),
          plot.title.position = "plot",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_blank(),
          axis.text.y = element_text(hjust = 0))
text.TN<-paste0("Probs - True Negative\n(NPV: ", round(cm$byClass["Neg Pred Value"],4), ")\n(Spec: ",round(cm$byClass["Specificity"],4),")")
plotpr.TN<-ggplot(data = shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="TN",], 
             aes(x = FS, y = text.TN)) +
                geom_boxplot(alpha=0.3) +
                xlab("") +
                ylab("") +
                labs(title = "") +
                scale_fill_discrete(guide="none") +
                # geom_vline(xintercept = 0) +
                theme_minimal() +
                theme(text = element_text(size = 15))

phi.FN<-round(colMeans(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FN",c(22:37)]),6)
val.FN<-mapply(get_max_prop_label, shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FN",c(1:16)], varname = covariates)
tittle.FN<-paste0("FN: False Negative (Pred: NS; Real: FS); n=", format(nrow(shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FN",]), big.mark = ","))
plotvs.FN<-ggplot(data.frame(var = covariates, phi = phi.FN, value = val.FN), 
      aes(x = reorder(value, abs(phi)),
          y=phi, 
          fill = ifelse(phi>0,'#FF0000','#0000FF'))) + 
  geom_bar(stat = "identity", alpha = 0.8) +
  # scale_fill_brewer(palette="Set1") +
  coord_flip() +
  labs(title = tittle.FN )+
  guides(fill = "none") +
  xlab("") +
  theme_minimal() +
  scale_y_continuous(name="Shapley Values",breaks = pretty(range(phi.FN), n = 5)) +
  scale_fill_manual(values=c("#FF0000" = "#FF0000", "#0000FF" = "#0000FF"))+ 
  scale_x_discrete(labels = function(x) parse(text = x)) +  
    theme(text = element_text(size = 15),
          plot.title = element_text(hjust = 0.5, size = 17),
          plot.title.position = "plot",
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          axis.line = element_blank(),
          axis.text.y = element_text(hjust = 0))
text.FN<-paste0("Probs - False Negative\n(NPV: ", round(cm$byClass["Neg Pred Value"],4), ")\n(Sens: ",round(cm$byClass["Sensitivity"],4),")")
plotpr.FN<-ggplot(data = shap_clusters_h_rf_under[shap_clusters_h_rf_under$CM=="FN",], 
             aes(x = FS, y = text.FN)) +
                geom_boxplot(alpha=0.3) +
                xlab("") +
                ylab("") +
                labs(title = "") +
                scale_fill_discrete(guide="none") +
                # geom_vline(xintercept = 0) +
                theme_minimal() +
                theme(text = element_text(size = 15))

plot.TP<-grid.arrange(plotvs.TP,plotpr.TP,nrow=2,heights = c(3.75, 1))
plot.FP<-grid.arrange(plotvs.FP,plotpr.FP,nrow=2,heights = c(3.75, 1))
plot.TN<-grid.arrange(plotvs.TN,plotpr.TN,nrow=2,heights = c(3.75, 1))
plot.FN<-grid.arrange(plotvs.FN,plotpr.FN,nrow=2,heights = c(3.75, 1))
g <- arrangeGrob(
  plot.TN, nullGrob(), plot.FN,
  nullGrob(), nullGrob(), nullGrob(),
  plot.FP, nullGrob(), plot.TP, 
  nrow = 3, ncol = 3,
  widths = c(1, 0.005, 1),
  heights = c(1, 0.075, 1)
)
g