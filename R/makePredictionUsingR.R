#' @title Perform state predictions from a specified model
#'
#' @description 
#' Calculates predictions
#' 
#' @param inputdata description
#' @param hyperdata description
#' @param parameters description
#' 
#' @export
makePredictionUsingR = function(inputdata,
                                ode.steps = 10,
                                ode.dt = diff(inputdata$t)/ode.steps,
                                ode.n = rep(ode.steps, nrow(inputdata)-1),
                                parvec = c(theta=10, mu=1, sigma_x=1, sigma_y=0.1),
                                X = 1,
                                P = 0.1
                                ){
  
  # Unload
  inputMat <- as.matrix(inputdata)
  parVec <- as.numeric(parvec)
  
  # Define functions
  f <- function(stateVec, parVec, inputVec){
    exp(parVec[1]) * (parVec[2] + inputVec[2] - stateVec[1])
  }
  dfdx <- function(stateVec, parVec, inputVec){
    -exp(parVec[1])
  }
  g <- function(stateVec, parVec, inputVec){
    exp(parVec[3])
  }
  
  # Storage
  N = nrow(inputMat)-1
  xPred = matrix(nrow=N+1,ncol=length(X))
  pPred = matrix(nrow=N+1,ncol=length(X)^2)
  xPred[1,] <- X
  pPred[1,] <- P 
  
  # For Loop
  # Solve moment (mean and variance) differential equations
  # with forward-euler method
  for(i in 1:N){
    
    inputVec <- as.numeric(inputMat[i,])
    dinputVec <- (inputMat[i,] - inputMat[i+1,])/ode.n[i]
    
    for(j in 1:ode.n[i]){
      F_ <- f(X, parVec, inputVec)
      A <- dfdx(X,parVec,inputVec)
      G <- g(X,parVec,inputVec)
      # 
      X <- X + F_ * ode.dt[i]
      # P <- P + ( A %*% P + P %*% t(A) + G %*% t(G)  ) * ode.dt[i]
      P <- P + ( 2*A*P + G^2  ) * ode.dt[i]
      # 
      inputVec <- inputVec + dinputVec
    }
  
    xPred[i+1,] = X
    pPred[i+1,] = P
  }
  
  return(list(xPred = xPred, pPred = pPred))
}
