% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.R
\name{geom_eval}
\alias{geom_eval}
\title{Adding Test Data Layer}
\usage{
geom_eval(data, colour = "red", num_train = 1, ...)
}
\arguments{
\item{data}{Test data to draw.}

\item{colour}{Colour of the line (By default, \code{"red"}).}

\item{num_train}{\ifelse{html}{\href{https://lifecycle.r-lib.org/articles/stages.html#experimental}{\figure{lifecycle-experimental.svg}{options: alt='[Experimental]'}}}{\strong{[Experimental]}} Size of the train data.}

\item{...}{Other arguments passed on the \code{\link[ggplot2:geom_path]{ggplot2::geom_path()}}.}
}
\description{
This function adds a layer of test dataset.
}