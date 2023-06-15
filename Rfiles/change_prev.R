# create plots to illustrate ability of ES data to detect a change in prevalence

library(tidyverse)
library(janitor)
library(ggplot2)
library(ggrepel)
library(RColorBrewer)
library(ggforce) # circle
library(cowplot)
library(survival)

setwd("~/GitHub/wbe_prevalence_england_python")

# load in data
# this is all ES data aggregated to a CIS region
dat <- read_csv("agg_data_inner_11_cis_mar22.csv")
head(dat)

table(dat$CIS20CD)

# this is the raw data with corresponding CIS for that sub-region
# key variables:
# - ww_site_code (ww site)
# - sars_cov2_gc_l_mean_v2 (corrected)
# - catchment_population_ons_mid_2019
# - CIS20CD
# - median_prob ll ul for CIS estimates
sat <- read_csv("raw_data_ww_10_cis_mar22.csv")
head(sat)
colnames(sat)

# sizes of CIS catchments
tmp <- sat %>% group_by(water_company_abbr,ww_site_code) %>% summarise(n=n(),
                                                    size=mean(catchment_population_ons_mid_2019))

# for sake of argument, focus on TW sites.
tdat <- sat %>% filter(water_company_abbr == "TW") %>% select(water_company_abbr,ww_site_code,
                                                              sars_cov2_gc_l_mean_v2,
                                                              CIS20CD,
                                                              median_prob,ll,ul,
                                                              date)
tdat$year_week <- paste0(format(tdat$date,"%Y"),"-",format(tdat$date,"%W"))
tdat$year_week_t1 <- (as.numeric(format(tdat$date,"%Y"))-2020)*17
tdat$year_week_num <- ifelse(tdat$year_week_t1==0,as.numeric(format(tdat$date,"%W"))-35,as.numeric(format(tdat$date,"%W"))+17)+1
table(tdat$year_week_num)

ggplot(tdat,aes(x=median_prob)) + geom_histogram()
ggplot(tdat,aes(x=date,y=median_prob)) + geom_point() + geom_line() +
  facet_wrap(~ww_site_code) + geom_hline(yintercept = 1.0,col="red")

# aggregate to week
tdatag <- tdat %>% group_by(water_company_abbr,ww_site_code,year_week,year_week_num,CIS20CD) %>% 
  summarise(n=n(),sars_cov2_gc_l_mean_v2_m = mean(sars_cov2_gc_l_mean_v2),
                  median_prob_m = mean(median_prob),
                  ll_m = mean(ll),
                  ul_m = mean(ul))
tdatag$ID <- 1:(dim(tdatag)[1])

tprev <- 1.0 # target prevalence
tdatag$target_prev <- 0
tdatag$target_prev[tdatag$ll_m<=tprev & tdatag$ul_m>=tprev] <- 1
table(tdatag$target_prev)

# create a case control dataset...
tdatag$controla <- tdatag$controlb <- NA
tdatag$case_plusa <- tdatag$case_plusb <- NA
tdatag$case_minusa <- tdatag$case_minusb <- NA
tdatag$match_id_1 <- NA
# control: target _prev = 1 AND in the next week no change in prevalence
# - loop through each target and see if 'not a target' the following week
vals <- which(tdatag$target_prev==1)
tdatag[vals[1],]

count_id_1 <- 1
for(i in 1:length(vals)){
  # vals == 18 tdatag[vals[i],]
  oo <- which(tdatag$ww_site_code == tdatag$ww_site_code[vals[i]] & tdatag$CIS20CD == tdatag$CIS20CD[vals[i]] & tdatag$year_week_num == tdatag$year_week_num[vals[i]]+1)
  if(length(oo)==1){
    # case or control?
    if((tdatag$ll_m[oo]<=tprev & tdatag$ul_m[oo]>=tprev)==TRUE){
      # control
      tdatag$controla[vals[i]] <- 1
      tdatag$controlb[oo] <- 1
      tdatag$match_id_1[vals[i]] <- count_id_1
      tdatag$match_id_1[oo] <- count_id_1
      count_id_1 <- count_id_1 + 1
    }
    if((tdatag$ll_m[oo]<=tprev & tdatag$ul_m[oo]<tprev)==TRUE){
      # case_minus
      tdatag$case_minusa[vals[i]] <- 1
      tdatag$case_minusb[oo] <- 1
      tdatag$match_id_1[vals[i]] <- count_id_1
      tdatag$match_id_1[oo] <- count_id_1
      count_id_1 <- count_id_1 + 1
    }
    if((tdatag$ll_m[oo]>tprev & tdatag$ul_m[oo]>=tprev)==TRUE){
      # case_plus
      tdatag$case_plusa[vals[i]] <- 1
      tdatag$case_plusb[oo] <- 1
      tdatag$match_id_1[vals[i]] <- count_id_1
      tdatag$match_id_1[oo] <- count_id_1
      count_id_1 <- count_id_1 + 1
    }
  }
  print(count_id_1)
}

table(tdatag$control)
table(tdatag$case_plus)
table(tdatag$case_minus)

# re-create the data
tmp <- tdatag %>% filter(controlb == 1)
dat <- tmp
tmp <- tdatag %>% filter(case_minusb == 1)
dat <- rbind(dat,tmp)
tmp <- tdatag %>% filter(case_plusb == 1)
dat <- rbind(dat,tmp)

dat$category <- NA
dat$category[dat$controlb == 1] <- "No change"
dat$category[dat$case_minusb == 1] <- "Reduction"
dat$category[dat$case_plusb == 1] <- "Increase"
dat$category <- factor(dat$category, levels = c("Reduction","No change",  "Increase"))

p1 <- ggplot(dat,aes(x=category,y=median_prob_m)) + geom_jitter() + geom_boxplot(alpha=0.5) +
  ylab("Prevalence from \nCovid Infection Survey") + xlab(" ") +
   ylim(0.4,2.6) + ggtitle(label="",subtitle = "'Gold Standard'")
p2 <- ggplot(dat,aes(x=category,y=sars_cov2_gc_l_mean_v2_m)) + 
  geom_jitter() + geom_boxplot(alpha=0.5) + scale_y_log10(limits=c(5e+03,3.5e+05)) +
  ylab("SARS-CoV-2 in ES (gc/l)") + xlab("Change in CIS prevalence the following week") +
  ggtitle(label="",subtitle = "P-value in log-linear model: \nreduction=0.375, increase<0.0001")

pdf("change_prev_initial_plot.pdf",height=5,width=4)
plot_grid(p1,p2,nrow=2)
dev.off()

dat$categoryB <- factor(dat$category, levels = c("No change", "Reduction", "Increase"))
m1 <- lm(log10(sars_cov2_gc_l_mean_v2_m) ~ categoryB, data = dat)
summary(m1)

# if we had a matched case-control...
# sars-cov-2 would be the variable, and others
head(dat)
# controlb == 1 == control
# case_plusb == 1 == case
# *** anyway the observations need to be matched really for this to be evaluated properly ***
# *** I have not done this...when I pick this up again I should do this ***
data <- dat %>% filter(category!="Reduction")
data$outcome <- 0
data$outcome[data$case_plusb==1] <- 1
table(data$match_id_1,data$outcome)
data$match_check <- 0
oo <- unique(data$match_id_1)
for(i in oo){
  # loop through each and check we have a 1:1 match
  aa <- which(data$match_id_1 == oo[i])
  if(length(aa)>1){
    if((data$case_plusb[aa]==1 & !is.na(data$case_plusb[aa])) == TRUE){
      
    }
  }
}


m2 <- clogit()

# end
