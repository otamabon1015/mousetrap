library(plyr)
library(mousetrap)
library(readbulk)
library(jsonlite)
library(tidyverse)
library(dataPreparation)



#### DATA LOADING UND MERGING #####

files = list.files("./RAW/parsed_data/", pattern="*.csv")
path = paste("./RAW/parsed_data/",files,sep="")

myfiles <- lapply(path,function(files){
  read.csv(files, header=T, skip=0,encoding="UTF-8", stringsAsFactors=F) 
})

train <- myfiles[[1]]
train$subject_nr <- 1
train$language <- myfiles[[1]][1, ]$language
train$age <- myfiles[[1]][2, ]$age
train$gender <- myfiles[[1]][2, ]$gender

for (i in 2:length(myfiles)) { 
  myfiles[[i]]$subject_nr <- i
  myfiles[[i]]$language <- myfiles[[i]][1, ]$language
  myfiles[[i]]$age <- myfiles[[i]][2, ]$age
  myfiles[[i]]$gender <- myfiles[[i]][2, ]$gender
  
  train <- merge(train, myfiles[[i]], all=TRUE)
}

View(train)


# Get rows with x and y coordinates and correct response
train <- 
  train %>% filter(sender =="present_stimulus" & !is.na(xpos) & !is.na(ypos) & !xpos == "" & !ypos == "" & correct == TRUE )

# Remove outliers
train <- remove_sd_outlier(train, cols = "duration", n_sigmas = 3, verbose = TRUE)

# Keep only relevant columns
raw<-train[,c(56, 5, 8, 18:24, 27:28, 32:36)]

# Check data
str(raw)
names(raw)
head(raw)
length(raw)

write.csv(raw,"RAW/preprocessed/preprocessed_data.csv", row.names = FALSE)

# Split data into training and test set
raw_train <- 
  raw %>% filter(subject_nr == 1  | subject_nr == 2 | subject_nr == 4 | subject_nr == 5 | subject_nr == 8 | subject_nr == 9 |
                   subject_nr == 10 | subject_nr == 12 | subject_nr == 14 | subject_nr == 15 | subject_nr == 16 | subject_nr == 17 |
                   subject_nr == 18 | subject_nr == 19 )

raw_test <- 
  raw %>% filter(subject_nr == 3  | subject_nr == 6 | subject_nr == 7 | subject_nr == 11 | subject_nr == 13 | subject_nr == 20)

# Save the data as csv file
write.csv(raw_train,"RAW/preprocessed/train/preprocessed_train_data.csv", row.names = FALSE)
write.csv(raw_test,"RAW/preprocessed/test/preprocessed_test_data.csv", row.names = FALSE)



#### DATA PREPROCESSING #####

raw_data <- read_opensesame("RAW/preprocessed/train") #"RAW/preprocessed/test" for extracting features from test data
mt_data <- mt_import_mousetrap(raw_data, unordered="remove")
mt_data <- mt_remap_symmetric(mt_data)
mt_data <- mt_align_start(mt_data, start=c(0,0))
mt_data <- mt_time_normalize(mt_data, nsteps = 100)

DL_mt_data <- mt_export_long(mt_data,
                             use=c("tn_trajectories"))

write.csv(DL_mt_data,"data/Stroop_DL_train_set.csv", row.names = FALSE)
write.csv(DL_mt_data,"data/Stroop_DL_test_set.csv", row.names = FALSE)

write.csv(raw_train$Condition,"data/Stroop_DL_train_set_Label.csv", row.names = FALSE)
write.csv(raw_test$Condition,"data/Stroop_DL_test_set_Label.csv", row.names = FALSE)



#### EXTRACTING FEATURES ####

mt_data <- mt_derivatives(mt_data) 
mt_data <- mt_deviations(mt_data) 
mt_data <- mt_measures(mt_data)

train_set <- dplyr::inner_join(
  mt_data$data, mt_data$measures, by="mt_id")

write.csv(train_set,"data/Stroop_train_set.csv", row.names = FALSE)
write.csv(train_set,"data/Stroop_test_set.csv", row.names = FALSE)

mt_plot_aggregate(mt_data, use="tn_trajectories", points=TRUE, 
                  x="xpos", y="ypos", color="Condition")


#### Descriptive Statistics ####


# Aggregate measures per condition
# separately per subject
average_measures <- mt_aggregate_per_subject(
  mt_data,
  use="measures",  use2_variables="Condition", subject_id="subject_nr"
)

mt_condition_1 <- average_measures %>% filter(Condition == "congruent")

mt_condition_2 <- average_measures %>% filter(Condition == "incongruent")

mt_data_statistics_1 <- mt_condition_1 %>% summarise(
  xpos_max_mean = mean(xpos_max),
  xpos_max_sd = sd(xpos_max),
  
  xpos_min_mean = mean(xpos_min),
  xpos_min_sd = sd(xpos_min),
  
  ypos_max_mean = mean(ypos_max),
  ypos_max_sd = sd(ypos_max),
  
  ypos_min_mean = mean(ypos_min),
  ypos_min_sd = sd(ypos_min),
  
  MAD_mean = mean(MAD),
  MAD_sd = sd(MAD),
  
  MAD_time_mean = mean(MAD_time),
  MAD_time_sd = sd(MAD_time),
  
  MD_above_mean = mean(MD_above),
  MD_above_sd = sd(MD_above),
  
  MD_above_time_mean = mean(MD_above_time),
  MD_above_time_sd = sd(MD_above_time),
  
  MD_below_mean = mean(MD_below),
  MD_below_sd = sd(MD_below),
  
  MD_below_time_mean = mean(MD_below_time),
  MD_below_time_sd = sd(MD_below_time),
  
  AD_mean = mean(AD),
  AD_sd = sd(AD),
  
  AUC_mean = mean(AUC),
  AUC_sd = sd(AUC),
  
  xpos_flips_mean = mean(xpos_flips),
  xpos_flips_sd = sd(xpos_flips),
  
  ypos_flips_mean = mean(ypos_flips),
  ypos_flips_sd = sd(ypos_flips),
  
  xpos_reversals_mean = mean(xpos_reversals),
  xpos_reversals_sd = sd(xpos_reversals),
  
  ypos_reversals_mean = mean(ypos_reversals),
  ypos_reversals_sd = sd(ypos_reversals),
  
  RT_mean = mean(RT),
  RT_sd = sd(RT),
  
  initiation_time_mean = mean(initiation_time),
  initiation_time_sd = sd(initiation_time),
  
  idle_time_mean = mean(idle_time),
  idle_time_sd = sd(idle_time),
  
  total_dist_mean = mean(total_dist),
  total_dist_sd = sd(total_dist),
  
  vel_max_mean = mean(vel_max),
  vel_max_sd = sd(vel_max),
  
  vel_max_time_mean = mean(vel_max_time),
  vel_max_time_sd = sd(vel_max_time),
  
  vel_min_mean = mean(vel_min),
  vel_min_sd = sd(vel_min),
  
  vel_min_time_mean = mean(vel_min_time),
  vel_min_time_sd = sd(vel_min_time),
  
  acc_max_mean = mean(acc_max),
  acc_max_sd = sd(acc_max),
  
  acc_max_time_mean = mean(acc_max_time),
  acc_max_time_sd = sd(acc_max_time),
  
  acc_min_mean = mean(acc_min),
  acc_min_sd = sd(acc_min),
  
  acc_min_time_mean = mean(acc_min_time),
  acc_min_time_sd = sd(acc_min_time)
)

mt_data_statistics_2 <- mt_condition_2 %>% summarise(
  xpos_max_mean = mean(xpos_max),
  xpos_max_sd = sd(xpos_max),
  
  xpos_min_mean = mean(xpos_min),
  xpos_min_sd = sd(xpos_min),
  
  ypos_max_mean = mean(ypos_max),
  ypos_max_sd = sd(ypos_max),
  
  ypos_min_mean = mean(ypos_min),
  ypos_min_sd = sd(ypos_min),
  
  MAD_mean = mean(MAD),
  MAD_sd = sd(MAD),
  
  MAD_time_mean = mean(MAD_time),
  MAD_time_sd = sd(MAD_time),
  
  MD_above_mean = mean(MD_above),
  MD_above_sd = sd(MD_above),
  
  MD_above_time_mean = mean(MD_above_time),
  MD_above_time_sd = sd(MD_above_time),
  
  MD_below_mean = mean(MD_below),
  MD_below_sd = sd(MD_below),
  
  MD_below_time_mean = mean(MD_below_time),
  MD_below_time_sd = sd(MD_below_time),
  
  AD_mean = mean(AD),
  AD_sd = sd(AD),
  
  AUC_mean = mean(AUC),
  AUC_sd = sd(AUC),
  
  xpos_flips_mean = mean(xpos_flips),
  xpos_flips_sd = sd(xpos_flips),
  
  ypos_flips_mean = mean(ypos_flips),
  ypos_flips_sd = sd(ypos_flips),
  
  xpos_reversals_mean = mean(xpos_reversals),
  xpos_reversals_sd = sd(xpos_reversals),
  
  ypos_reversals_mean = mean(ypos_reversals),
  ypos_reversals_sd = sd(ypos_reversals),
  
  RT_mean = mean(RT),
  RT_sd = sd(RT),
  
  initiation_time_mean = mean(initiation_time),
  initiation_time_sd = sd(initiation_time),
  
  idle_time_mean = mean(idle_time),
  idle_time_sd = sd(idle_time),
  
  total_dist_mean = mean(total_dist),
  total_dist_sd = sd(total_dist),
  
  vel_max_mean = mean(vel_max),
  vel_max_sd = sd(vel_max),
  
  vel_max_time_mean = mean(vel_max_time),
  vel_max_time_sd = sd(vel_max_time),
  
  vel_min_mean = mean(vel_min),
  vel_min_sd = sd(vel_min),
  
  vel_min_time_mean = mean(vel_min_time),
  vel_min_time_sd = sd(vel_min_time),
  
  acc_max_mean = mean(acc_max),
  acc_max_sd = sd(acc_max),
  
  acc_max_time_mean = mean(acc_max_time),
  acc_max_time_sd = sd(acc_max_time),
  
  acc_min_mean = mean(acc_min),
  acc_min_sd = sd(acc_min),
  
  acc_min_time_mean = mean(acc_min_time),
  acc_min_time_sd = sd(acc_min_time)
)
