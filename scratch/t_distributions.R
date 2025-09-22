library(tidyverse)

dt_var1 <- function(x, df) {
  scale <- sqrt((df - 2) / df)   # rescales variance to 1
  dt(x / scale, df = df) / scale
}

extent <- 3.5 # x range
dfs <- c(4, 8, 32) # degrees of freedom to plot

expand.grid(
    x = seq(-extent, extent, length.out = 500),
    df = dfs
  ) %>%
  mutate(density = map2_dbl(x, df, ~dt_var1(.x, .y))) %>%
  ggplot(aes(x = x, y = density, color = factor(df))) +
  geom_line(size = 1) +
  labs(x = "x",
       y = "Density",
       color = "Degrees of Freedom",
       title = "Rescaled t-distributions") +
  theme_minimal()
