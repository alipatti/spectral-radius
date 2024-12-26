library(tidyr)
library(dplyr)
library(purrr)
library(forcats)
library(gssr)
library(gssrdoc)

data(gss_all)
data(gss_dict)

load_or_create <- function(path, creation_fn) {
    if (file.exists(path)) {
        readr::read_rds(path)
    } else {
        df <- rlang::as_function(creation_fn)()
        rlang::write_rds(df, path)
        df
    }
}

gss_variables <- load_or_create("data/gss_variables.rds", ~ {
    raw <- 1:7 %>%
        purrr::map(~ {
            # backend for https://gssdataexplorer.norc.org/variables/vfilter
            gss_explorer_api_url <- paste0(
                "https://3ilfsaj2lj.execute-api.us-east-1.amazonaws.com",
                "/prod/variables/guest-search"
            )

            # not positive that all these fields are necessary
            body <- list(
                page = .x,
                limit = 1000,
                subjects = NULL,
                modules = NULL,
                my_tags = NULL,
                shared_tags = NULL,
                yearRange = c(1972, 2022),
                years = NULL,
                parameter = NULL,
                term = NULL,
                workspaceId = NULL,
                yearBallot = NULL
            ) %>%
                jsonlite::toJSON(auto_unbox = T, null = "null", )

            httr::POST(gss_explorer_api_url, body = body) %>%
                httr::content("text", encoding = "utf-8") %>%
                jsonlite::fromJSON(flatten = T) %>%
                pluck("variables") %>%
                as_tibble()
        }, .progress = TRUE) %>%
        bind_rows()

    clean <- raw %>%
        janitor::clean_names() %>%
        unnest(survey_question, keep_empty = TRUE) %>%
        # unnest_wider(module, names_sep = "_")
        select(-tag_info)

    clean
})

# TODO: find subset of variables that are representitive
# and asked across wide swath of years
variables <- c(
    # "homosex", "spkhomo", "colhomo",
    # "libhomo", "helpblk", "discaff",
    "affrmact", "fejobaff",
    "discaff"
    # "discaffm",
    # "discaffw"
)

df <-
    gss_all %>%
    select(year, any_of(variables)) %>%
    mutate(
        across(
            all_of(variables),
            . %>%
                haven::zap_missing() %>%
                as_factor() %>%
                factor(., levels = levels(.), ordered = T)
        ),
        across(c(year), as.integer)
    ) %>%
    drop_na()
