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
estimateParameters = function(obsdata,
                              inputdata,
                              x0 = 3,
                              p0 = 0.1*diag(1),
                              ode.dt = diff(inputdata$t)/10,
                              ode.n = rep(10, nrow(inputdata)-1),
                              parameters,
                              parameter_map = NULL,
                              lb = -Inf,
                              ub = Inf,
                              silent = TRUE,
                              trace = 0,
                              control = list(trace=trace, eval.max=1e4, iter.max=1e4),
                              hessian = TRUE){
  
  
  tmbdata = list(
    
    # initial
    stateVec = x0,
    covMat = p0,
    
    # ODE
    ode_timestep_size = ode.dt,
    ode_timesteps = ode.n,
    ode_solver = 2,
    
    # loss function
    loss_function = 0,
    loss_threshold_value = 3,
    tukey_loss_parameters = 1:4,
    
    # estimate stationary levels
    estimate_stationary_initials = 0,
    initial_variance_scaling = 1,
    
    # map
    MAP_bool = 0,
    
    # inputs
    inputMat = as.matrix(inputdata),
    
    # observations
    obsMat = as.matrix(obsdata)
    
  )
  
  ##### CONSTRUCT NEG. LOG-LIKELIHOOD  #####
  nll = TMB::MakeADFun(tmbdata, 
                       parameters,
                       DLL = "test_model", 
                       map = parameter_map, 
                       silent = silent)
  
  ##### OPTIMIZE  #####
  if (hessian) {
    opt = try(stats::nlminb(nll$par, 
                            nll$fn, 
                            nll$gr, 
                            nll$he, 
                            lower=lb, 
                            upper=ub,
                            control=control), 
              silent=T)
  } else {
    opt <- try(stats::nlminb(nll$par, 
                             nll$fn, 
                             nll$gr, 
                             lower=lb, 
                             upper=ub,
                             control=control),
               silent=T)
  }
  
  ##### RETURN  #####
  if (inherits(opt,"try-error")) {
    opt = list()
    # opt$par = parameters$par
  }
  
  
  return(list(opt=opt,nll=nll))
}
