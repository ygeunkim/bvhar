% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/summary-forecast.R
\name{rmape}
\alias{rmape}
\alias{rmape.predbvhar}
\alias{rmape.bvharcv}
\title{Evaluate the Model Based on RMAPE (Relative MAPE)}
\usage{
rmape(x, pred_bench, y, ...)

\method{rmape}{predbvhar}(x, pred_bench, y, ...)

\method{rmape}{bvharcv}(x, pred_bench, y, ...)
}
\arguments{
\item{x}{Forecasting object to use}

\item{pred_bench}{The same forecasting object from benchmark model}

\item{y}{Test data to be compared. should be the same format with the train data.}

\item{...}{not used}
}
\value{
RMAPE vector corresponding to each variable.
}
\description{
This function computes RMAPE given prediction result versus evaluation set.
}
\details{
RMAPE is the ratio of MAPE of given model and the benchmark one.
Let \eqn{MAPE_b} be the MAPE of the benchmark model.
Then

\deqn{RMAPE = \frac{mean(MAPE)}{mean(MAPE_b)}}

where \eqn{MAPE} is the MAPE of our model.
}
\references{
Hyndman, R. J., & Koehler, A. B. (2006). \emph{Another look at measures of forecast accuracy}. International Journal of Forecasting, 22(4), 679-688.
}
