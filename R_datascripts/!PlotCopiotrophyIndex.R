# Load necessary libraries
library(dplyr)
library(ggplot2)
library(ggsignif)
library(tibble)
library(FSA)
library(ggpubr)
library(tidyr)
library(forcats)
library(grid)
library(scales) 
library(ggmap)
library(maps)
library(mapdata)
### ------------------- Start from here -------------------------------------------------------------------------------
setwd("YOURPATH")
table <- read.csv("DenitrifierCopiotrophyTable.csv", header = TRUE) 
RA <- read.csv("DenitrifierRA.csv", header = TRUE)
tablewithRA <- merge(table, RA, by="Genome")
table <- tablewithRA

# Assign type_DIN based on functional_type_order
table$type_DIN <- ifelse(table$functional_type %in% c("nitrate1", "nitrate2", "nitrate3"), "nitrate",
                         ifelse(table$functional_type %in% c("nitrite1", "nitrite2"), "nitrite",
                                ifelse(table$functional_type == "N2O1", "N2O", NA)))
new_labels <- c(
  "nitrate1" = expression(NO[3]^"-"~"→"~NO[2]^"-"),
  "nitrate2" = expression(NO[3]^"-"~"→"~N[2]*O),
  "nitrate3" = expression(NO[3]^"-"~"→"~N[2]),
  "nitrite1" = expression(NO[2]^"-"~"→"~N[2]*O),
  "nitrite2" = expression(NO[2]^"-"~"→"~N[2]),
  #"NitrateN2O" = "Bookend",
  #"Other" = "Other",
  "N2O1" = expression(N[2]*O~"→"~N[2]))

# Define the desired order
functional_type_order <- c("nitrate1", "nitrate2", "nitrate3",
                           "nitrite1", "nitrite2", "N2O1")#,
                           #"NitrateN2O", "Other")


# Convert functional_type into a factor with specified order
table$functional_type <- factor(table$functional_type, levels = functional_type_order)
# Categorize type_group
table$type_group <- ifelse(table$type %in% c("MAG", "SAG"), "MAG/SAG", "Genome")

table_isolates <- table %>% filter(type == "genome")
table_MAGsSAGs <- table %>% filter(type != "genome")
table_MAGsSAGs_highQ <- table_MAGsSAGs %>% filter(Completeness >= 90, Contamination <= 5)
head(table)


#####--------------Datasets counts---------
# make labels for legend match your other plot
table$type_group <- factor(table$type_group,
                           levels = c("Genome", "MAG/SAG"),
                           labels = c("Isolates", "MAGs+SAGs"))

ggplot(table, aes(x = functional_type, fill = type_group)) +
  geom_bar(position = "stack", width = 0.8, alpha = 0.9) +
  labs(title = "", x = "", y = "Count", fill = "") +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 14, face = "bold", angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = c(0.98, 0.98),
    legend.justification = c(1, 1),
    legend.text = element_text(size = 14, face = "bold"),
    legend.title = element_text(size = 14, face = "bold"),
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line.x = element_line(color = "black", linewidth = 0.5),
    axis.line.y = element_line(color = "black", linewidth = 0.5),
    axis.ticks.length = unit(-0.25, "cm"),
    axis.ticks.x = element_blank()
  ) +
  scale_fill_manual(values = c("Isolates" = "royalblue",
                               "MAGs+SAGs" = "goldenrod2")) +
  scale_x_discrete(
    drop = FALSE,
    labels = function(x) lapply(x, function(y) new_labels[[y]])
  )

ggsave("FigS2_datasets_Count.png", width = 18, height = 12, units = "cm")


################--------------------------------need to run steps 1 - 3 for later analyses!-----------------------------------------------------
# Step 1: Add Source Label to Each Dataset
table_isolates <- table_isolates %>% mutate(Source = "Isolates")
table_MAGsSAGs <- table_MAGsSAGs %>% mutate(Source = "MAGs+SAGs")

# Step 2: Merge Data
table_combined <- bind_rows(table_isolates, table_MAGsSAGs)

# Step 3: Convert functional_type and Source into Factors for Proper Ordering
table_combined$functional_type <- factor(table_combined$functional_type, levels = functional_type_order)
table_combined$Source <- factor(table_combined$Source, levels = c("Isolates", "MAGs+SAGs"))

# Fig 3a
ggplot(table_combined, aes(x = Source, y = IoC, fill = Source)) + 
  geom_violin(alpha = 0.6, width = 1.25) +
  geom_jitter(aes(color = Source), width = 0.2, size = 1.5, alpha = 0.7) +
  stat_summary(fun = median, geom = "point", color = "firebrick", size = 2, stroke = 1.5, shape = 4) +
  stat_summary(fun = mean, geom = "point", color = "firebrick", size = 3, shape = 16) + 
  theme_minimal() +
  labs(title = "", x = "", y = "IoC") +
  theme_bw() +
  theme(axis.text.x = element_text(size = 14, face = "bold"), 
        axis.text.y = element_text(size = 12, face = "bold"), 
        axis.title.x = element_text(size = 14, face = "bold"), 
        axis.title.y = element_text(size = 14, face = "bold"), 
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5), 
        legend.position='none', # Hides redundant legend
        panel.border = element_blank(), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(), 
        axis.line.x = element_line(color = "black", size = 0.5), 
        axis.line.y = element_line(color = "black", size = 0.5), 
        axis.ticks.length = unit(-0.25, "cm")) +
  scale_fill_manual(values = c("Isolates" = "royalblue", "MAGs+SAGs" = "goldenrod2")) +
  scale_color_manual(values = c("Isolates" = "royalblue", "MAGs+SAGs" = "goldenrod2")) 

ggsave("Fig3a_CompareGenomeToDraftalltypetogether_IoC_violin.png", width = 10, height = 12, units = "cm")


##########################-----------------Relative abundance Plots ------------ -------------------------------------------

table <- table_combined# 
# Convert AMAL columns to long format for processing
AMAL_columns <- grep("^AMAL\\d+$", colnames(table), value = TRUE)

# Compute sum of AMAL for each Source and AMAL
summary_table <- table %>%
  pivot_longer(cols = all_of(AMAL_columns), names_to = "AMAL", values_to = "AMAL_value") %>%
  group_by(Source, AMAL) %>%
  summarise(total_AMAL = sum(AMAL_value, na.rm = TRUE), .groups = "drop")

# Read the AMAL location table
location_data <- read.csv("/Users/xinsun/Library/CloudStorage/Dropbox/!Carnegie/!Research/!Manuscript_Den&SNM&N2O/3.CopioOligo-lifestyle/RA_fromIrene/AMAL_location_table.csv", header = TRUE)


# Merge the summed AMAL data with location data
merged_data <- summary_table %>%
  inner_join(location_data, by = c("AMAL" = "AMALname")) %>%
  mutate(AMAL_label = paste(Location, depth, "m", sep = " - "))  # Create new label

# Order AMAL labels by Location first, then by Depth (shallowest first)
merged_data <- merged_data %>%
  arrange(Location, depth) %>%
  mutate(AMAL_label = factor(AMAL_label, levels = unique(AMAL_label)))  # Ordered factor


# Fig 3b
filtered_data <- merged_data %>% filter(O2 <= 50)

ggplot(merged_data, aes(x = Source, y = total_AMAL, fill = Source)) +
  geom_violin(alpha = 0.6, width = 1.25) +
  #geom_boxplot(alpha = 0.6, width = 1.25) +
  geom_jitter(aes(color = Source), width = 0.2, size = 1.5, alpha = 0.7) +
  stat_summary(fun = median, geom = "point", color = "firebrick", size = 2, stroke = 1.5, shape = 4) +
  stat_summary(fun = mean, geom = "point", color = "firebrick", size = 3, shape = 16) +
  theme_minimal() +
  labs(title = "", x = "", y = "Relative abundance (%)") +
  theme_bw() +
  theme(
    axis.text.x = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = 'none',
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line.x = element_line(color = "black", size = 0.5),
    axis.line.y = element_line(color = "black", size = 0.5),
    axis.ticks.length = unit(-0.25, "cm")
  ) +
  scale_fill_manual(values = c("Isolates" = "royalblue", "MAGs+SAGs" = "goldenrod2")) +
  scale_color_manual(values = c("Isolates" = "royalblue", "MAGs+SAGs" = "goldenrod2"))

# Save figure
ggsave("Fig3b_CompareGenomeToDraftalltypetogether_RA_violin.png", width = 10, height = 12, units = "cm")


##########################-----------------Weighted IoC with Relative abundance Plots ------------ -------------------------------------------
table <- table_MAGsSAGs #table_combined #  table_isolates # 

# Convert AMAL columns to long format for processing
AMAL_columns <- grep("^AMAL\\d+$", colnames(table), value = TRUE)

#### Choose 1 of the 2 below! 
#1. Compute weighted IoC for each functional type and AMAL
summary_table <- table %>%
  pivot_longer(cols = all_of(AMAL_columns), names_to = "AMAL", values_to = "AMAL_value") %>%
  group_by(functional_type, AMAL) %>%
  summarise(weighted_IoC = sum(IoC * AMAL_value, na.rm = TRUE) / sum(AMAL_value, na.rm = TRUE), .groups = "drop")
#2.  or compute weighted dCUB, nCAZy, nGenes
summary_table <- table %>%
 pivot_longer(cols = all_of(AMAL_columns), names_to = "AMAL", values_to = "AMAL_value") %>%
 group_by(functional_type, AMAL) %>%
  #summarise(weighted_IoC = sum(nGenes * AMAL_value, na.rm = TRUE) / sum(AMAL_value, na.rm = TRUE), .groups = "drop")
  #summarise(weighted_IoC = sum(nCAZy * AMAL_value, na.rm = TRUE) / sum(AMAL_value, na.rm = TRUE), .groups = "drop")
  summarise(weighted_IoC = sum(dCUB * AMAL_value, na.rm = TRUE) / sum(AMAL_value, na.rm = TRUE), .groups = "drop")


# Merge the weighted IoC data with location data
merged_data <- summary_table %>%
  inner_join(location_data, by = c("AMAL" = "AMALname")) %>%
  mutate(AMAL_label = paste(Location, depth, "m", sep = " "))  # Create new label

# Order labels by location first, then by depth (shallowest first)
merged_data <- merged_data %>%
  arrange(Location, depth) %>%
  mutate(AMAL_label = factor(AMAL_label, levels = unique(AMAL_label)))  # Ordered factor

# Create text labels for O2 and NO2
merged_data <- merged_data %>%
  mutate(label_text = paste0("O2: ", round(O2, 2), "\nNO2: ", round(NO2, 2)))  # Format label text

# Create text labels for O2 and NO2
merged_data <- merged_data %>%
  mutate(label_text = paste0(round(O2, 1)))  # Format label text

## Fig 4 
ggplot(merged_data, aes(x = functional_type, y = weighted_IoC)) +
  geom_violin(fill = "goldenrod2", width = 1.2, color = alpha("black", 0.6), alpha = 0.6) +#
  geom_jitter(aes(color = O2), width = 0.2, size = 2, alpha = 0.8) +
  # Median point
  stat_summary(
    fun = median,
    geom = "point", aes(group = functional_type),
    color = "firebrick", size = 2, stroke = 1.5, shape = 4,
    position = position_dodge(width = 0.8)
  ) +
  # Mean point
  stat_summary(
    fun = mean,
    geom = "point", aes(group = functional_type),
    color = "firebrick", size = 3, shape = 16,
    position = position_dodge(width = 0.8)
  ) +
  # Color scale for O2
  scale_color_viridis_c(option = "plasma") +
  # Optional: distinct fill colors per functional_type (just for violin)
  scale_fill_manual(values = scales::hue_pal()(length(unique(merged_data$functional_type)))) +
  labs(
    title = "",
    x = "",
    y = "Weighted IoC",#"Weighted nGenes", #"Weighted dCUB",#
    color = "O2 (μM)"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 14, face = "bold"),
    axis.text.y = element_text(size = 12, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.title.y = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
    legend.position = "right",
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line.x = element_line(color = "black", size = 0.5),
    axis.line.y = element_line(color = "black", size = 0.5),
    axis.ticks.length = unit(-0.25, "cm"),
    axis.ticks.x = element_blank()
  ) +
  #scale_y_continuous(trans = "log10", labels = scales::comma) + 
  scale_y_reverse() +
  scale_x_discrete(labels = function(x) lapply(x, function(y) new_labels[[y]]))

ggsave("Fig4b_WeightedIoC.png", width = 14, height = 12, units = "cm") 
#ggsave("Fig4c_WeighteddCUB.png", width = 14, height = 12, units = "cm") 
#ggsave("Fig4e_WeightednGenes.png", width = 14, height = 12, units = "cm") 


## Fig 5
labeler_parsed_custom <- labeller(functional_type = as_labeller(new_labels, default = label_parsed))
filtered_data <- merged_data %>% filter(O2 <= 50)

ggplot(filtered_data, aes(x = depth, y = weighted_IoC, color = Location)) +
  geom_point(shape = 1, size = 2, stroke = 2, alpha = 1) +
  geom_line(aes(group = Location), linewidth = 1.5, alpha = 0.7) +
  #facet_wrap(functional_type ~ Location, scales = "free", labeller = labeler_parsed_custom) + #scales = "free_y",
  facet_grid(rows = vars(functional_type), cols = vars(Location), scales = "free", labeller = labeler_parsed_custom) +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "",
    x = "Depth (m)",
    y = "Weighted IoC",
    color = "Location"
  ) +
  scale_x_reverse() +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 14, face = "bold"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 14),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey90"),
    axis.line = element_line(color = "black", linewidth = 0.4)
  )

ggsave("Fig5_WeightedIoC_depths.png", width = 20, height = 24, units = "cm") 




