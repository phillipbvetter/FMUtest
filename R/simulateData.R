#' @title Estimate parameters in SDE model.
#' 
#' @description 
#' Maximum likelihood based estimation of state space parameters.
#' 
#' @param obsdata description
#' @param inputdata description
#' @param hyperdata description
#' @param parameters description
#' @param lb description
#' @param ub description
#' @param silent description
#' @param hessian description
#'
#' 
#' @export
simulateData = function(pars = c(theta=10, mu=1, sigma_x=1, sigma_y=0.1),
                        seed = runif(1)*1e4,
                        dt.sim = 1e-3,
                        dt.obs = 1e-1,
                        x0 = 3
                        ){
  
  # Simulate data using Euler Maruyama
  set.seed(seed)
  pars = c(theta=10, mu=1, sigma_x=1, sigma_y=0.1)
  
  # Simulate
  dt.sim = dt.sim
  t.sim = seq(0,5,by=dt.sim)
  dw = rnorm(length(t.sim)-1,sd=sqrt(dt.sim))
  u.sim = cumsum(rnorm(length(t.sim),sd=0.05))
  x = x0
  for(i in 1:(length(t.sim)-1)) {
    x[i+1] = x[i] + pars[1]*(pars[2]-x[i]+u.sim[i])*dt.sim + pars[3]*dw[i]
  }
  
  # Extract observations and add noise
  dt.obs = dt.obs
  ids = seq(1,length(t.sim),by=round(dt.obs / dt.sim))
  t.obs = t.sim[ids]
  y = x[ids] + pars[4] * rnorm(length(t.obs))
  u = u.sim[ids]
  
  # Create data
  .data = data.frame(
    t = t.obs,
    y = y,
    u = u
  )
  
  return(.data)
  
}
