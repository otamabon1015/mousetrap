library(plyr)
library(mousetrap)
library(readbulk)
library(jsonlite)
library(tidyverse)
library(dataPreparation)


#### DATA LOADING UND MERGING #####

#source("./r-jatos-json-parser/parseJSONdata.R")

#result = "./RAW/participant_24"
#parseJSONdata(result)

# Load data
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

# Get rows with x and y coordinates
train <- 
  train %>% filter(sender =="present_stimulus" & !is.na(xpos) & !is.na(ypos) & !xpos == "" & !ypos == ""  )

# Remove outliers
train <- remove_sd_outlier(train, cols = "duration", n_sigmas = 3, verbose = TRUE)

# Keep only relevant columns
raw<-train[,c(56, 5, 8, 18:24, 27:28, 32:36)]

# Map German and Japanese exemplar words to English
raw$Exemplar <- mapvalues(raw$Exemplar, 
                          from=c("blau","rot","gelb","grün","lila","青","赤","黄色","緑","紫"), 
                          to=c("blue","red","yellow","green","purple","blue","red","yellow","green","purple"))

# Check data
str(raw)
names(raw)


########## DATA CHECKING ###########
# Participants
xtabs(~ subject_nr, raw)
# Count
xtabs(~ count, raw)
# Condition
xtabs(~ Condition, raw)
# Probanden + Condition
xtabs(~subject_nr + Condition, raw)
# Participants + Exemplars + Condition
xtabs(~Exemplar + subject_nr + Condition, raw)

# Save participants table as object
objekt1 <- xtabs(~ subject_nr, raw)
objekt1
str(objekt1)

# Save participants table as dataframe
objekt1 <- as.data.frame(xtabs(~ subject_nr, raw))
objekt1
str(objekt1)

# Save participants and condition table as dataframe
objekt2 <- as.data.frame(xtabs(~ subject_nr + Condition, raw))
objekt2
str(objekt2)

# Check variance and mean of each column
str(raw)
summary(raw)

# Convert the variable "correct" to factor type
raw$correct<-as.factor(raw$correct)

# Boxplots for reaction time
boxplot(raw$duration)

# Boxplots for reaction time (for each participant)
boxplot(raw$duration ~ raw$subject_nr)
boxplot(raw$duration ~ raw$Exemplar)

summary(raw$correct)
xtabs(~ + correct + Condition, raw)



# Now keep only rows with correct response
raw <- 
  raw %>% filter(!is.na(xpos) & !is.na(ypos) & !xpos == "" & !ypos == "" & correct == TRUE)

# Reaction time
# Grand Means
ddply(raw, .(Condition), summarize, durchschnitt= mean(duration),SD=sd(duration))
# Mean per Participants
subagg_datacorr2 <- ddply(datacorr, .(Condition, subject_nr), summarize, durchschnitt= mean(duration),SD=sd(duration))
subagg_datacorr2

# Mean of mean per Participants
subagg_dtcorr_mean2 <- ddply(subagg_datacorr2, .(Condition), summarize, Mittel= mean(durchschnitt),SD=sd(durchschnitt))
subagg_dtcorr_mean2

# Mean per Items
subitm_datacorr2 <- ddply(datacorr, .(Condition, Exemplar), summarize, durchschnitt= mean(duration),SD=sd(duration))
subitm_datacorr2
# Mean of mean per Items
subitm_dtcorr_mean2 <- ddply(subitm_datacorr2, .(Condition), summarize, Mittel= mean(durchschnitt),SD=sd(durchschnitt))
subitm_dtcorr_mean2