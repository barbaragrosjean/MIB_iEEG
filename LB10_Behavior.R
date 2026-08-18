############################################################
# BEHAVIORAL ANALYSES
#
# Data columns:
#
# recognition_onset
# keypress_onset
# RT
# subject
# ev_id
# cond
# success
#
# Analyses:
#
# 1. Accuracy above chance (50%)
#    -> Binomial GLMM
#
# 2. Accuracy by condition
#    -> Binomial GLMM
#
# 3. RT by condition
#    -> Linear Mixed-Effects Model (LMM)
############################################################


############################################################
# 0. PACKAGES
############################################################

packages <- c(
  "tidyverse",
  "lme4",
  "lmerTest",
  "emmeans"
)

installed <- rownames(installed.packages())

for (p in packages) {
  if (!(p %in% installed)) {
    install.packages(p)
  }
}

library(tidyverse)
library(lme4)
library(lmerTest)
library(emmeans)


############################################################
# 1. LOAD DATA
############################################################

# Replace with your actual file name
data <- read.csv("your_data.csv")


############################################################
# 2. CHECK DATA
############################################################

str(data)

summary(data)

head(data)


############################################################
# 3. PREPARE VARIABLES
############################################################

data <- data %>%
  mutate(
    subject = factor(subject),
    ev_id = factor(ev_id),
    cond = factor(
      cond,
      levels = c("Memorized", "Novel")
    ),
    success = as.numeric(success),
    RT = as.numeric(RT)
  )


############################################################
# 4. CHECK CODING
############################################################

cat("\nCondition counts:\n")
print(table(data$cond, useNA = "ifany"))

cat("\nAccuracy counts:\n")
print(table(data$success, useNA = "ifany"))

cat("\nNumber of participants:\n")
print(n_distinct(data$subject))

cat("\nMissing values:\n")
print(colSums(is.na(data)))


############################################################
# 5. CALCULATE RT IF NECESSARY
############################################################

# If RT is ALREADY correctly calculated, you do NOT need
# to run this section.
#
# If RT is not calculated, uncomment the following line:
#
# data <- data %>%
#   mutate(RT = keypress_onset - recognition_onset)


############################################################
# ANALYSIS 1
#
# ACCURACY ABOVE CHANCE (50%)
#
# Binomial GLMM
#
# Model:
#
# success ~ 1 + (1 | subject)
#
# H0: accuracy = 50%
# H1: accuracy > 50%
#
# On the logit scale:
#
# 50% = logit(0.5) = 0
############################################################


accuracy_data <- data %>%
  filter(
    !is.na(subject),
    !is.na(success)
  )


# Fit GLMM
model_accuracy_chance <- glmer(
  success ~ 1 + (1 | subject),
  data = accuracy_data,
  family = binomial(link = "logit")
)


# Model summary
summary(model_accuracy_chance)


############################################################
# Extract intercept
############################################################

coef_chance <- summary(
  model_accuracy_chance
)$coefficients["(Intercept)", ]

beta_chance <- coef_chance["Estimate"]

SE_chance <- coef_chance["Std. Error"]

z_chance <- coef_chance["z value"]


############################################################
# One-sided test:
# Is accuracy > 50%?
############################################################

# beta = 0 corresponds to 50% accuracy

p_one_sided_chance <- pnorm(
  z_chance,
  lower.tail = FALSE
)


############################################################
# Odds ratio
############################################################

OR_chance <- exp(beta_chance)


############################################################
# 95% CI for OR
############################################################

CI_beta_chance <- confint(
  model_accuracy_chance,
  parm = "(Intercept)",
  method = "Wald"
)

OR_CI_chance <- exp(CI_beta_chance)


############################################################
# Estimated accuracy from GLMM
############################################################

estimated_accuracy <- plogis(beta_chance)


############################################################
# Descriptive statistics
############################################################

M_accuracy <- mean(
  accuracy_data$success,
  na.rm = TRUE
)

SD_accuracy <- sd(
  accuracy_data$success,
  na.rm = TRUE
)

N_trials <- nrow(accuracy_data)

N_subjects <- n_distinct(
  accuracy_data$subject
)


############################################################
# PRINT ANALYSIS 1
############################################################

cat("\n")
cat("====================================================\n")
cat("ANALYSIS 1: ACCURACY ABOVE CHANCE\n")
cat("====================================================\n")

cat(
  "Mean accuracy = ",
  round(M_accuracy, 3),
  "\n"
)

cat(
  "SD = ",
  round(SD_accuracy, 3),
  "\n"
)

cat(
  "N trials = ",
  N_trials,
  "\n"
)

cat(
  "N participants = ",
  N_subjects,
  "\n"
)

cat(
  "Estimated GLMM accuracy = ",
  round(estimated_accuracy, 3),
  "\n"
)

cat(
  "Log-odds intercept = ",
  round(beta_chance, 3),
  "\n"
)

cat(
  "SE = ",
  round(SE_chance, 3),
  "\n"
)

cat(
  "OR = ",
  round(OR_chance, 3),
  "\n"
)

cat(
  "95% CI for OR = [",
  round(OR_CI_chance[1], 3),
  ", ",
  round(OR_CI_chance[2], 3),
  "]\n",
  sep = ""
)

cat(
  "z = ",
  round(z_chance, 3),
  "\n"
)

cat(
  "One-sided p = ",
  format.pval(
    p_one_sided_chance,
    digits = 4
  ),
  "\n"
)


############################################################
# ANALYSIS 2
#
# ACCURACY BY CONDITION
#
# Model:
#
# success ~ cond + (1 | subject)
#
# Memorized = reference
# Novel = comparison
############################################################


accuracy_condition_data <- data %>%
  filter(
    !is.na(subject),
    !is.na(cond),
    !is.na(success)
  )


############################################################
# Fit GLMM
############################################################

model_accuracy_condition <- glmer(
  success ~ cond + (1 | subject),
  data = accuracy_condition_data,
  family = binomial(link = "logit")
)


############################################################
# Model summary
############################################################

summary(model_accuracy_condition)


############################################################
# Extract Novel effect
############################################################

coef_novel <- summary(
  model_accuracy_condition
)$coefficients["condNovel", ]

beta_novel <- coef_novel["Estimate"]

SE_novel <- coef_novel["Std. Error"]

z_novel <- coef_novel["z value"]

p_novel <- coef_novel["Pr(>|z|)"]


############################################################
# Odds ratio
############################################################

OR_novel <- exp(beta_novel)


############################################################
# 95% CI
############################################################

CI_beta_novel <- confint(
  model_accuracy_condition,
  parm = "condNovel",
  method = "Wald"
)

OR_CI_novel <- exp(CI_beta_novel)


############################################################
# Estimated accuracy by condition
############################################################

accuracy_emmeans <- emmeans(
  model_accuracy_condition,
  ~ cond,
  type = "response"
)

print(accuracy_emmeans)


############################################################
# Pairwise comparison
############################################################

accuracy_pairwise <- contrast(
  accuracy_emmeans,
  method = "pairwise",
  adjust = "none"
)

print(accuracy_pairwise)


############################################################
# Descriptive accuracy by condition
############################################################

accuracy_descriptives <- accuracy_condition_data %>%
  group_by(cond) %>%
  summarise(
    N_trials = n(),
    N_participants = n_distinct(subject),
    M_accuracy = mean(success, na.rm = TRUE),
    SD_accuracy = sd(success, na.rm = TRUE),
    .groups = "drop"
  )

print(accuracy_descriptives)


############################################################
# PRINT ANALYSIS 2
############################################################

cat("\n")
cat("====================================================\n")
cat("ANALYSIS 2: ACCURACY BY CONDITION\n")
cat("====================================================\n")

cat(
  "Novel vs Memorized OR = ",
  round(OR_novel, 3),
  "\n"
)

cat(
  "95% CI = [",
  round(OR_CI_novel[1], 3),
  ", ",
  round(OR_CI_novel[2], 3),
  "]\n",
  sep = ""
)

cat(
  "z = ",
  round(z_novel, 3),
  "\n"
)

cat(
  "p = ",
  format.pval(
    p_novel,
    digits = 4
  ),
  "\n"
)

cat("\nDescriptive accuracy:\n")

print(accuracy_descriptives)


############################################################
# ANALYSIS 3
#
# RESPONSE TIME BY CONDITION
#
# Linear Mixed-Effects Model
#
# Model:
#
# RT ~ cond + (1 | subject)
#
# Memorized = reference
# Novel = comparison
############################################################


############################################################
# Select valid RT trials
############################################################

RT_data <- data %>%
  filter(
    !is.na(subject),
    !is.na(cond),
    !is.na(RT),
    RT > 0
  )


############################################################
# Descriptive RT
############################################################

RT_descriptives <- RT_data %>%
  group_by(cond) %>%
  summarise(
    N_trials = n(),
    N_participants = n_distinct(subject),
    M_RT = mean(RT, na.rm = TRUE),
    SD_RT = sd(RT, na.rm = TRUE),
    .groups = "drop"
  )

print(RT_descriptives)


############################################################
# Fit LMM
############################################################

model_RT <- lmer(
  RT ~ cond + (1 | subject),
  data = RT_data,
  REML = TRUE
)


############################################################
# Model summary
############################################################

summary(model_RT)


############################################################
# Extract Novel effect
############################################################

coef_RT <- summary(
  model_RT
)$coefficients["condNovel", ]

beta_RT <- coef_RT["Estimate"]

SE_RT <- coef_RT["Std. Error"]

df_RT <- coef_RT["df"]

t_RT <- coef_RT["t value"]

p_RT <- coef_RT["Pr(>|t|)"]


############################################################
# 95% CI
############################################################

CI_RT <- confint(
  model_RT,
  parm = "condNovel",
  method = "Wald"
)


############################################################
# Estimated RT by condition
############################################################

RT_emmeans <- emmeans(
  model_RT,
  ~ cond
)

print(RT_emmeans)


############################################################
# Pairwise comparison
############################################################

RT_pairwise <- contrast(
  RT_emmeans,
  method = "pairwise",
  adjust = "none"
)

print(RT_pairwise)


############################################################
# PRINT ANALYSIS 3
############################################################

cat("\n")
cat("====================================================\n")
cat("ANALYSIS 3: RESPONSE TIME BY CONDITION\n")
cat("====================================================\n")

cat(
  "Memorized mean RT = ",
  round(
    RT_descriptives$M_RT[
      RT_descriptives$cond == "Memorized"
    ],
    2
  ),
  "\n"
)

cat(
  "Novel mean RT = ",
  round(
    RT_descriptives$M_RT[
      RT_descriptives$cond == "Novel"
    ],
    2
  ),
  "\n"
)

cat(
  "Novel - Memorized difference = ",
  round(beta_RT, 2),
  "\n"
)

cat(
  "95% CI = [",
  round(CI_RT[1], 2),
  ", ",
  round(CI_RT[2], 2),
  "]\n",
  sep = ""
)

cat(
  "t = ",
  round(t_RT, 3),
  "\n"
)

cat(
  "df = ",
  round(df_RT, 2),
  "\n"
)

cat(
  "p = ",
  format.pval(
    p_RT,
    digits = 4
  ),
  "\n"
)


############################################################
# OPTIONAL: RT DISTRIBUTION
############################################################

ggplot(
  RT_data,
  aes(x = RT)
) +
  geom_histogram(
    bins = 50
  ) +
  labs(
    title = "Response Time Distribution",
    x = "Response Time",
    y = "Frequency"
  ) +
  theme_minimal()


############################################################
# OPTIONAL: RT BY CONDITION
############################################################

ggplot(
  RT_data,
  aes(
    x = cond,
    y = RT
  )
) +
  geom_boxplot() +
  labs(
    title = "Response Time by Condition",
    x = "Condition",
    y = "Response Time"
  ) +
  theme_minimal()


############################################################
# OPTIONAL: CHECK RT SKEW
############################################################

hist(
  RT_data$RT,
  breaks = 50,
  main = "RT Distribution",
  xlab = "Response Time"
)


############################################################
# OPTIONAL: LOG-RT LMM
#
# If RT is strongly right-skewed, consider this model.
############################################################

RT_data <- RT_data %>%
  mutate(
    log_RT = log(RT)
  )

model_log_RT <- lmer(
  log_RT ~ cond + (1 | subject),
  data = RT_data,
  REML = TRUE
)

summary(model_log_RT)


############################################################
# OPTIONAL: SAVE RESULTS
############################################################

results <- data.frame(

  analysis = c(
    "Accuracy above chance",
    "Accuracy: Novel vs Memorized",
    "RT: Novel vs Memorized"
  ),

  estimate = c(
    OR_chance,
    OR_novel,
    beta_RT
  ),

  CI_lower = c(
    OR_CI_chance[1],
    OR_CI_novel[1],
    CI_RT[1]
  ),

  CI_upper = c(
    OR_CI_chance[2],
    OR_CI_novel[2],
    CI_RT[2]
  ),

  statistic = c(
    z_chance,
    z_novel,
    t_RT
  ),

  p_value = c(
    p_one_sided_chance,
    p_novel,
    p_RT
  )
)

print(results)

write.csv(
  results,
  "behavioral_analysis_results.csv",
  row.names = FALSE
)