###############################################
# 0. Setup
###############################################
library(tidyverse)
library(lubridate)

# >>>>> CHANGE THIS IF YOUR FOLDER MOVES <<<<<
data_dir <- "/Users/elizabethrice/Desktop/Data Analytics Labs/DataScience/project/data"

###############################################
# 1. Load CSVs from your data folder
###############################################

games <- read.csv(file.path(data_dir, "games.csv"))
games_weather <- read.csv(file.path(data_dir, "games_weather.csv"))
stadiums <- read.csv(file.path(data_dir, "stadium_coordinates.csv"))
teamrankings <- read.csv(file.path(data_dir, "teamrankings.csv"))

# If anything errors here, check the filenames exactly in Finder.

###############################################
# 2. Build Bills home games + weather dataset
###############################################

# Stadium(s) where BUF is the home team
bills_stadiums <- stadiums %>%
  filter(HomeTeam == "BUF") %>%
  pull(StadiumName) %>%
  unique()

print(bills_stadiums)  # just to confirm

# Check some column names so we know what we’re working with
# (uncomment the next lines once to see structure)
# colnames(games)
# head(games$TimeStartGame)

# Filter to games in Bills home stadium
bills_games <- games %>%
  filter(StadiumName %in% bills_stadiums) %>%
  mutate(
    # TimeStartGame in the WeatherData repo looks like "9/3/2000 20:20"
    game_datetime = mdy_hm(TimeStartGame),
    game_date = as.Date(game_datetime)
  )

# Join to games_weather hourly data
bills_games_weather <- bills_games %>%
  inner_join(games_weather, by = "game_id")

# Summarise to one row per game with average conditions
bills_games_summary <- bills_games_weather %>%
  group_by(game_id, Season, StadiumName, game_date) %>%
  summarise(
    mean_temp    = mean(Temperature,   na.rm = TRUE),
    min_temp     = min(Temperature,    na.rm = TRUE),
    max_temp     = max(Temperature,    na.rm = TRUE),
    mean_wind    = mean(WindSpeed,     na.rm = TRUE),
    mean_precip  = mean(Precipitation, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    cold_game = if_else(mean_temp <= 32, "Cold", "Not cold")
  )

###############################################
# 3. Clean TeamRankings data robustly
###############################################

# See what your columns are actually called:
print(colnames(teamrankings))

# Try to auto-detect the Win% column (something with "win" and "%" or "pct")
win_cols <- names(teamrankings)[
  grepl("win", names(teamrankings), ignore.case = TRUE) &
    grepl("%|pct|perc", names(teamrankings), ignore.case = TRUE)
]

win_col <- win_cols[1]
cat("Using Win% column:", win_col, "\n")

# Try to auto-detect MOV / margin column
mov_cols <- names(teamrankings)[
  grepl("mov|margin", names(teamrankings), ignore.case = TRUE)
]
mov_col <- mov_cols[1]
cat("Using MOV column:", mov_col, "\n")

teamrankings_clean <- teamrankings %>%
  mutate(
    WinPct_raw     = .data[[win_col]],
    WinPct_numeric = as.numeric(gsub("%", "", WinPct_raw)),
    MOV_numeric    = if (!is.na(mov_col)) as.numeric(.data[[mov_col]]) else NA_real_
  ) %>%
  filter(!is.na(WinPct_numeric))

###############################################
# 4. PLOTS — 10 graphs
###############################################

########## PLOT 1 (Data Description)
# Histogram of average game temperature
ggplot(bills_games_summary, aes(x = mean_temp)) +
  geom_histogram(binwidth = 5) +
  labs(
    title = "Distribution of Average Game-Time Temperature\nBuffalo Bills Home Games (2000–2020)",
    x = "Average Game Temperature (°F)",
    y = "Number of Games"
  ) +
  theme_minimal()

########## PLOT 2 (Data Description)
# Number of home games per season
games_per_season <- bills_games_summary %>%
  group_by(Season) %>%
  summarise(n_games = n(), .groups = "drop")

ggplot(games_per_season, aes(x = factor(Season), y = n_games)) +
  geom_col() +
  labs(
    title = "Number of Buffalo Bills Home Games per Season",
    x = "Season",
    y = "Number of Home Games"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

########## PLOT 3 (Exploratory)
# Boxplot of mean temperature by season
ggplot(bills_games_summary,
       aes(x = factor(Season), y = mean_temp)) +
  geom_boxplot() +
  labs(
    title = "Average Game-Time Temperature by Season\nBuffalo Bills Home Games",
    x = "Season",
    y = "Average Game Temperature (°F)"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

########## PLOT 4 (Exploratory)
# Cold vs non-cold overall counts
cold_counts <- bills_games_summary %>%
  group_by(cold_game) %>%
  summarise(n_games = n(), .groups = "drop")

ggplot(cold_counts,
       aes(x = cold_game, y = n_games)) +
  geom_col() +
  labs(
    title = "Cold vs Non-Cold Bills Home Games\n(Threshold: 32°F)",
    x = "Game Type",
    y = "Number of Games"
  ) +
  theme_minimal()

########## PLOT 5 (Exploratory)
# Cold vs non-cold per season (stacked bars)
cold_counts_season <- bills_games_summary %>%
  group_by(Season, cold_game) %>%
  summarise(n_games = n(), .groups = "drop")

ggplot(cold_counts_season,
       aes(x = factor(Season), y = n_games, fill = cold_game)) +
  geom_col() +
  labs(
    title = "Cold vs Non-Cold Bills Home Games by Season",
    x = "Season",
    y = "Number of Games",
    fill = "Game Type"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

########## PLOT 6 (Exploratory)
# Temp vs wind, colored by cold_game
ggplot(bills_games_summary,
       aes(x = mean_temp, y = mean_wind, color = cold_game)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Average Wind Speed vs Temperature\nBuffalo Bills Home Games",
    x = "Average Game Temperature (°F)",
    y = "Average Wind Speed (mph)",
    color = "Game Type"
  ) +
  theme_minimal()

########## PLOT 7 (Exploratory)
# Temp vs precipitation, colored by cold_game
ggplot(bills_games_summary,
       aes(x = mean_temp, y = mean_precip, color = cold_game)) +
  geom_point() +
  geom_smooth(method = "loess", se = TRUE) +
  labs(
    title = "Average Precipitation vs Temperature\nBuffalo Bills Home Games",
    x = "Average Game Temperature (°F)",
    y = "Average Precipitation (inches)",
    color = "Game Type"
  ) +
  theme_minimal()

########## PLOT 8 (Exploratory)
# Boxplot: wind speed by cold vs not cold
ggplot(bills_games_summary,
       aes(x = cold_game, y = mean_wind)) +
  geom_boxplot() +
  labs(
    title = "Wind Speed Distribution in Cold vs Non-Cold Games",
    x = "Game Type",
    y = "Average Wind Speed (mph)"
  ) +
  theme_minimal()

########## PLOT 9 (Exploratory)
# Density of average game temperature by cold/not-cold

ggplot(bills_games_summary,
       aes(x = mean_temp, fill = cold_game)) +
  geom_density(alpha = 0.6) +
  labs(
    title = "Temperature Distributions for Cold vs Non-Cold Bills Home Games",
    x = "Average Game Temperature (°F)",
    y = "Density",
    fill = "Game Type"
  ) +
  theme_minimal()


########## PLOT 10 (Exploratory)
# Average of average game temperature per season (trend)

temp_trend_season <- bills_games_summary %>%
  group_by(Season) %>%
  summarise(
    avg_season_temp = mean(mean_temp, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(temp_trend_season,
       aes(x = Season, y = avg_season_temp)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Average Game-Time Temperature by Season\nBuffalo Bills Home Games",
    x = "Season",
    y = "Average Game Temperature (°F)"
  ) +
  theme_minimal()
