% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/simulateData.R
\name{simulateData}
\alias{simulateData}
\title{Estimate parameters in SDE model.}
\usage{
simulateData(
  pars = c(theta = 10, mu = 1, sigma_x = 1, sigma_y = 0.1),
  seed = runif(1) * 10000,
  dt.sim = 0.001,
  dt.obs = 0.1,
  x0 = 3
)
}
\arguments{
\item{obsdata}{description}

\item{inputdata}{description}

\item{hyperdata}{description}

\item{parameters}{description}

\item{lb}{description}

\item{ub}{description}

\item{silent}{description}

\item{hessian}{description}
}
\description{
Maximum likelihood based estimation of state space parameters.
}
