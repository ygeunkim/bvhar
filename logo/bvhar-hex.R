library(tidyverse)
library(hexSticker)
# Subplot--------------------------------------
img_plt <-
  tibble(
    x = 1:5,
    y1 = c(0, 1, .5, 2, 1.5),
    y2 = c(1, 2, 1.5, 3, 2.5)
  ) |>
  pivot_longer(-x, names_to = "series", values_to = "value") |>
  mutate(
    ymin = case_when(
      x == 3 ~ value,
      x > 3 ~ value - .2,
      .default = NA
    ),
    ymax = case_when(
      x == 3 ~ value,
      x > 3 ~ value + .2,
      .default = NA
    )
  ) |>
  ggplot(aes(x = x, y = value)) +
  geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = series), alpha = .9, col = NA, na.rm = TRUE) +
  scale_fill_manual(values = c("#F5ECD5", "#D0DDD0")) +
  geom_path(aes(color = series)) +
  scale_color_manual(values = c("#D8C4B6", "#3E5879")) +
  guides(color = "none", fill = "none") +
  theme_void() +
  theme_transparent()
# Save sticker---------------------------------
sysfonts::font_add_google("Noto Sans")
sticker(
  # Subplot
  subplot = img_plt,
  s_x = 1,
  s_y = .9,
  s_width = 1.5,
  s_height = 1.5 * .618,
  # Package name
  package = c("b", "vhar"),
  p_family = "Noto Sans",
  p_fontface = "plain",
  p_size = 12,
  p_x = c(.5, 1.1),
  p_y = 1.25,
  p_color = c("#F39E60", "#F0BB78"),
  # Hexagon
  h_size = 1.2,
  h_fill = "#FFF0DC",
  h_color = "#131010",
  # Spotlight
  spotlight = FALSE,
  l_x = 1,
  l_y = 0.5,
  l_width = 3,
  l_height = 3,
  l_alpha = 0.4,
  # URL
  url = "ygeunkim.github.io/package/bvhar",
  u_x = 1,
  u_y = 0.08,
  u_color = "black",
  u_family = "Noto Sans",
  u_size = 1,
  u_angle = 30,
  # Save
  white_around_sticker = FALSE,
  filename = "./logo/bvhar-logo.png",
  asp = 1,
  dpi = 300
)
# usethis--------------------------------------
usethis::use_logo("./logo/bvhar-logo.png")
pkgdown::build_favicons(pkg = ".", overwrite = FALSE)
