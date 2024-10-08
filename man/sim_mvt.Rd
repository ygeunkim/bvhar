% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generate-process.R
\name{sim_mvt}
\alias{sim_mvt}
\title{Generate Multivariate t Random Vector}
\usage{
sim_mvt(num_sim, df, mu, sig, method = c("eigen", "chol"))
}
\arguments{
\item{num_sim}{Number to generate process.}

\item{df}{Degrees of freedom.}

\item{mu}{Location vector}

\item{sig}{Scale matrix.}

\item{method}{Method to compute \eqn{\Sigma^{1/2}}.
Choose between \code{eigen} (spectral decomposition) and \code{chol} (cholesky decomposition).
By default, \code{eigen}.}
}
\value{
T x k matrix
}
\description{
This function samples n x multi-dimensional t-random matrix.
}
