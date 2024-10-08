% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/plot.R
\name{autoplot.bvharirf}
\alias{autoplot.bvharirf}
\title{Plot Impulse Responses}
\usage{
\method{autoplot}{bvharirf}(object, ...)
}
\arguments{
\item{object}{A \code{bvharirf} object}

\item{...}{Other arguments passed on the \code{\link[ggplot2:geom_path]{ggplot2::geom_path()}}.}
}
\value{
A ggplot object
}
\description{
Draw impulse responses of response ~ impulse in the facet.
}
\seealso{
\code{\link[=irf]{irf()}}
}
