n <- 1e3
x <- rnorm(n)
t <- rbinom(n, 1, 0.5)
model <- \(.x, .t) 10 + 2 * .x - 1 * .t
y <- 10 + 2 * x + 1 * t + rnorm(n, sd = 0.1)
print(cor(model(x, t), y))
print(lm(model(x, t) ~ x + t))
print(lm(y ~ x + t))


library(tidyverse)
theme_set(theme_classic(base_size = 16))
x <- rnorm(2e3)
t <- rbinom(2e3, 1, 0.5)
y <- x + t + rnorm(2e3, sd = .5)

ggplot(tibble(x = x, y = y, t = ifelse(t == 1, "treated", "control")), aes(x, y, color = t)) +
    geom_point(size = 1.5) +
    labs(
        x = "Predicted",
        y = "Observed"
    ) +
    geom_abline(intercept = 0, linetype = 2, linewidth = 1.5, slope = 1, color = "red") +
    theme(
        legend.position = "inside",
        legend.position.inside = c(.8, .2),
        legend.text = element_text(size = 16)
    ) +
    coord_cartesian(xlim = c(-3, 3), ylim = c(-3, 3)) +
    scale_color_brewer(palette = "Dark2", name = NULL)
ggsave(
    "scratch.png",
    width = 6,
    height = 4.5,
    dpi = 300
)
d %>%
    ggplot(aes(factor(t), y = y)) +
    geom_violin()
ggplot(tibble(x = rnorm(2e3), y = x + rnorm(2e3, sd = 1.5)), aes(x, y)) +
    geom_point(size = 1.5) +
    labs(
        x = "Predicted",
        y = "Observed"
    ) +
    geom_abline(intercept = 0, linetype = 2, linewidth = 1.5, slope = 1, color = "red") +
    theme(legend.position = "none") +
    coord_cartesian(xlim = c(-3, 3), ylim = c(-3, 3))
ggsave(
    "scratch2.png",
    width = 6,
    height = 4.5,
    dpi = 300
)
