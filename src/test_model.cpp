#include <TMB.hpp>
using namespace density;

//////////// FIND NA INDICES IN VECTOR ///////////
  template <class Type>
  vector<Type> is_not_na(vector<Type> x){
    vector<Type> y(x.size());
    y.fill(Type(1.0));
      for(int i=0; i<x.size(); i++){
        if( R_IsNA(asDouble(x(i))) ){
          y(i) = Type(0.0);
        }
      }
    return y;
  }

//////////// REMOVE NA'S FROM VECTOR ///////////
  template<class Type>
  vector<Type> remove_nas__(vector<Type> obsVec, int number_of_nonNA_observations, vector<Type> is_not_na_vector){
    int ii = 0;
    vector<Type> y_reduced(number_of_nonNA_observations);
      for(int i=0; i < obsVec.size(); i++){
        if(is_not_na_vector(i) == Type(1.0)){
          y_reduced(ii) = obsVec(i);
          ii++;
        }
      }
    return y_reduced;
  }

//////////// helper fun: construct permutation matrix ///////////
  template <class Type>
  matrix<Type> construct_permutation_matrix(int number_of_nonNA_observations, int number_of_obs_eqs, vector<Type> is_not_na_vector){
    matrix<Type> E(number_of_nonNA_observations, number_of_obs_eqs);
    E.setZero();
    int j=0;
    for(int i=0; i < number_of_obs_eqs; i++){
      /*if p(i) is 1 then include by setting 1 in diagonal of matrix*/
      if(is_not_na_vector(i) == Type(1.0)){
        E(j,i) = Type(1.0);
        j += 1;
      }
    }
    return E;
  }

//////////// loss function ///////////
  template<class Type>
  Type lossfunction__(Type x, vector<Type> tukeypars, Type huber_c, int lossFunc){
    Type loss;
    if(lossFunc==1){
      Type a = tukeypars(0);
      Type b = tukeypars(1);
      Type c = tukeypars(2);
      Type d = tukeypars(3);
      loss = d * ( (Type(1.0)/(Type(1.0)+exp(-a*(x-b)))) + c );
    } else if (lossFunc==2){
      Type c_squared = pow(huber_c,2);
      loss = c_squared * (sqrt(1 + (x / c_squared)) - 1);
    } else {
      loss = x;
    }
    return loss;
  }

//////////// MAP estimation helper ///////////
  template<class Type>
  vector<Type> get_free_pars__(vector<int> mapints, int sum_mapints, vector<Type> parvec) {
    vector<Type> ans(sum_mapints);
    int j=0;
    for(int i=0;i<mapints.size();i++){
      if(mapints(i)==1){
        ans(j) = parvec(i);
        j += 1;
      }
    }
    return ans;
  }

//////////// drift function //////////
  template<class Type>
  vector<Type> f__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    vector<Type> f__(1);
    f__(0) = exp(parVec(0)) * (parVec(1) + inputVec(1) - stateVec(0));
    return f__;
  }

//////////// jacobian of drift function ///////////
  template<class Type>
  matrix<Type> dfdx__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    matrix<Type> dfdx__(1, 1);
    dfdx__(0, 0) = -exp(parVec(0));
    return dfdx__;
  }

//////////// diffusion function ///////////
  template<class Type>
  matrix<Type> g__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    matrix<Type> g__(1, 1);
    g__(0, 0) = exp(parVec(2));
    return g__;
  }

//////////// observation function ///////////
  template<class Type>
  vector<Type> h__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    vector<Type> h__(1);
    h__(0) = stateVec(0);
    return h__;
  }

//////////// jacobian of obs function ///////////
  template<class Type>
  matrix<Type> dhdx__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    matrix<Type> dhdx__(1, 1);
    dhdx__(0, 0) = 1;
    return dhdx__;
  }

//////////// observation variance matrix function ///////////
  template<class Type>
  vector<Type> hvar__(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    vector<Type> hvar__(1);
    hvar__(0) = pow(exp(parVec(3)), 2);
    return hvar__;
  }

//////////// 1-step f moment ODE ///////////
  template<class Type>
  matrix<Type> cov_ode_1step(matrix<Type> covMat, vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    matrix<Type> cov_ode_1step = dfdx__(stateVec, parVec, inputVec) * covMat + covMat * dfdx__(stateVec, parVec, inputVec).transpose() + g__(stateVec, parVec, inputVec) * g__(stateVec, parVec, inputVec).transpose();
    return cov_ode_1step;
  }

//////////// ODE SOLVER ///////////
  template<class Type>
  struct ode_integration {
  
    vector<Type> X1;
    matrix<Type> P1;
  
    ode_integration(matrix<Type> covMat, vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec, vector<Type> dinputVec, Type dt, int ode_solver){
      /*Initial State and Cov Values*/
      vector<Type> X0 = stateVec;
      matrix<Type> P0 = covMat;
  
      /*Forward Euler*/
      if(ode_solver == 1){
  
       X1 = X0 + f__(stateVec, parVec, inputVec) * dt;
       P1 = P0 + cov_ode_1step(covMat, stateVec, parVec, inputVec) * dt;
  
      /*4th Order Runge-Kutta 4th*/
      } else if (ode_solver == 2){
  
       vector<Type> k1, k2, k3, k4;
       matrix<Type> c1, c2, c3, c4;
  
       /*1. Approx Slope at Initial Point*/
       k1 = f__(stateVec, parVec, inputVec);
       c1 = cov_ode_1step(covMat, stateVec, parVec, inputVec);
  
       /*2. First Approx Slope at Midpoint*/
       inputVec += 0.5 * dinputVec;
       stateVec = X0 + 0.5 * dt * k1;
       covMat   = P0 + 0.5 * dt * c1;
       k2       = f__(stateVec, parVec, inputVec); 
       c2       = cov_ode_1step(covMat, stateVec, parVec, inputVec);        
  
       /*3. Second Approx Slope at Midpoint*/
       stateVec = X0 + 0.5 * dt * k2;
       covMat   = P0 + 0.5 * dt * c2;
       k3       = f__(stateVec, parVec, inputVec); 
       c3       = cov_ode_1step(covMat, stateVec, parVec, inputVec);
  
       /*4. Approx Slope at End Point*/
       inputVec += 0.5 * dinputVec;
       stateVec = X0 + dt * k3;
       covMat   = P0 + dt * c3;
       k4       = f__(stateVec, parVec, inputVec); 
       c4       = cov_ode_1step(covMat, stateVec, parVec, inputVec);
  
       /*ODE UPDATE*/
       X1 = X0 + dt * (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
       P1 = P0 + dt * (c1 + 2.0*c2 + 2.0*c3 + c4)/6.0;
  
     } else {
     /*nothing*/
     }
    }
  };

 ////// Newton Solver for Stationary Mean //////
  template<class Type>
  struct Functor {
    vector<Type> p; // parVec
    vector<Type> u; // Inputs
    Functor(const vector<Type> &p, const vector<Type> &u) : p(p), u(u) {}
    Type operator()(const vector<Type> &s) {
      vector<Type> f = f__(s, p, u);
      return((f * f).sum());
    }
  };

 ////// Lyapunov Solve for Stationary Variance (Linearized System) ///////
  template<class Type>
  matrix<Type> LyapSolver(vector<Type> stateVec, vector<Type> parVec, vector<Type> inputVec){
    int n = stateVec.size();
    matrix<Type> I(n,n);
    I.setIdentity();
    matrix<Type> A = dfdx__(stateVec, parVec, inputVec);
    matrix<Type> GGT = g__(stateVec, parVec, inputVec) * g__(stateVec, parVec, inputVec).transpose();
    matrix<Type> GGT1d = GGT.vec().matrix();
    /* Kronecker */ 
    matrix<Type> X = tmbutils::kronecker(I, A) + tmbutils::kronecker(A, I);
    /*Solve Linear Systems */
    matrix<Type> P0 = Type(-1.0) * X.householderQr().solve(GGT1d).reshaped(n,n);
    /* Return */  
    return P0;
  }
template<class Type>
Type objective_function<Type>::operator() ()
{
DATA_INTEGER(ode_solver);
Type nll__ = 0;

//// observations ////
DATA_MATRIX(obsMat)

//// inputs ////
DATA_MATRIX(inputMat)

//// initial state ////
DATA_VECTOR(stateVec);
DATA_MATRIX(covMat);
DATA_STRUCT(cfg, newton::newton_config_t);
DATA_VECTOR(ode_timestep_size);
DATA_IVECTOR(ode_timesteps);

//// loss parameters ////
DATA_VECTOR(tukey_loss_parameters);
DATA_INTEGER(loss_function);
DATA_SCALAR(loss_threshold_value);

//// map estimation ////
DATA_INTEGER(MAP_bool);

//// parameters ////
PARAMETER(logtheta);
PARAMETER(mu);
PARAMETER(logsigma_x);
PARAMETER(logsigma_y);

//// system size ////
int number_of_state_eqs = 1;
int number_of_obs_eqs = 1;
int tsize = inputMat.col(0).size();

//// state, par, input, obs vectors ////
vector<Type> inputVec(2), dinputVec(2), obsVec(1), parVec(4);
parVec << logtheta, mu, logsigma_x, logsigma_y;

 //////////// initialize variables ///////////
int number_of_nonNA_observations;
Type half_log2PI = Type(0.5) * log(2*M_PI);
vector<Type> is_not_na_obsVec, e__, y__, F__, H__;
matrix<Type> C__, R__, K__, E__, V__, Ri__, A__, G__;

//////////// identity matrix ///////////
matrix<Type> I__(number_of_state_eqs, number_of_state_eqs), V0__(number_of_obs_eqs, number_of_obs_eqs);
I__.setIdentity();
V0__.setZero();

//////////// storage variables ///////////
vector<vector<Type>> xPrior(tsize), xPost(tsize), Innovation(tsize);
vector<matrix<Type>> pPrior(tsize), pPost(tsize), InnovationCovariance(tsize);
vector<Type> nll_report(tsize);

//////////// stationary mean ///////////
DATA_INTEGER(estimate_stationary_initials);
DATA_SCALAR(initial_variance_scaling);
if(estimate_stationary_initials == 1){
inputVec = inputMat.row(0);
Functor<TMBad::ad_aug> F(parVec, inputVec);
stateVec = newton::Newton(F, stateVec, cfg);
covMat = LyapSolver(stateVec, parVec, inputVec) * initial_variance_scaling;
};

//////////// set initial value ///////////
xPrior(0) = stateVec;
pPrior(0) = covMat;

 //////////// THE FIRST POINT DATA-UPDATE ///////////
obsVec = obsMat.row(0);
is_not_na_obsVec = is_not_na(obsVec);
number_of_nonNA_observations = CppAD::Integer(sum(is_not_na_obsVec));
if( number_of_nonNA_observations > 0 ){
inputVec = inputMat.row(0);
y__ = remove_nas__(obsVec, number_of_nonNA_observations, is_not_na_obsVec);
E__ = construct_permutation_matrix(number_of_nonNA_observations, number_of_obs_eqs, is_not_na_obsVec);
H__ = h__(stateVec, parVec, inputVec);
C__ = E__ * dhdx__(stateVec, parVec, inputVec);
e__ = y__ - E__ * H__;
V0__.diagonal() << hvar__(stateVec, parVec, inputVec);
V__ = E__ * V0__ * E__.transpose();
R__ = C__ * covMat * C__.transpose() + V__;
Ri__ = R__.inverse();
K__ = covMat * C__.transpose() * Ri__;
// State update // 
stateVec = stateVec + K__*e__;
covMat = (I__ - K__ * C__) * covMat * (I__ - K__ * C__).transpose() + K__* V__ * K__.transpose();
// Likelihood contribution // 
nll_report(0) = Type(0.5) * atomic::logdet(R__) + Type(0.5) * lossfunction__((e__*(Ri__*e__)).sum(), tukey_loss_parameters, loss_threshold_value, loss_function) + half_log2PI * asDouble(number_of_nonNA_observations);
nll__ += nll_report(0);
// Save residual mean/covariance // 
Innovation(0) = e__;
InnovationCovariance(0) = R__;
}
xPost(0) = stateVec;
pPost(0) = covMat;

 //////////// START MAIN LOOP ///////////
for(int i=0 ; i < tsize - 1 ; i++){
inputVec = inputMat.row(i);
dinputVec = (inputMat.row(i+1) - inputMat.row(i))/ode_timesteps(i);

 //////////// TIME-UPDATE: SOLVE MOMENT ODES ///////////
for(int j=0 ; j < ode_timesteps(i) ; j++){
ode_integration<Type> odelist = {covMat, stateVec, parVec, inputVec, dinputVec, ode_timestep_size(i), ode_solver};
stateVec = odelist.X1;
covMat = odelist.P1;
inputVec += dinputVec;
}
xPrior(i+1) = stateVec;
pPrior(i+1) = covMat;

 //////////// DATA-UPDATE ///////////
obsVec = obsMat.row(i+1);
is_not_na_obsVec = is_not_na(obsVec);
number_of_nonNA_observations = CppAD::Integer(sum(is_not_na_obsVec));
if( number_of_nonNA_observations > 0 ){
inputVec = inputMat.row(i+1);
y__ = remove_nas__(obsVec, number_of_nonNA_observations, is_not_na_obsVec);
E__ = construct_permutation_matrix(number_of_nonNA_observations, number_of_obs_eqs, is_not_na_obsVec);
H__ = h__(stateVec, parVec, inputVec);
C__ = E__ * dhdx__(stateVec, parVec, inputVec);
e__ = y__ - E__ * H__;
V0__.diagonal() << hvar__(stateVec, parVec, inputVec);
V__ = E__ * V0__ * E__.transpose();
R__ = C__ * covMat * C__.transpose() + V__;
Ri__ = R__.inverse();
K__ = covMat * C__.transpose() * Ri__;
// State update // 
stateVec = stateVec + K__*e__;
covMat = (I__ - K__ * C__) * covMat * (I__ - K__ * C__).transpose() + K__* V__ * K__.transpose();
// Likelihood contribution // 
nll_report(i+1) = Type(0.5) * atomic::logdet(R__) + Type(0.5) * lossfunction__((e__*(Ri__*e__)).sum(), tukey_loss_parameters, loss_threshold_value, loss_function) + half_log2PI * asDouble(number_of_nonNA_observations);
nll__ += nll_report(i+1);
// Save residual mean/covariance // 
Innovation(i+1) = e__;
InnovationCovariance(i+1) = R__;
}
xPost(i+1) = stateVec;
pPost(i+1) = covMat;
}
//////////// END MAIN LOOP ///////////

//////////// MAP CONTRIBUTION ///////////
if(MAP_bool == 1){
DATA_VECTOR(map_mean__);
DATA_MATRIX(map_cov__);
DATA_IVECTOR(map_ints__);
DATA_INTEGER(sum_map_ints__);
vector<Type> map_pars__;
map_pars__ = get_free_pars__(map_ints__, sum_map_ints__, parVec);
vector<Type> pars_eps__ = map_pars__ - map_mean__;
matrix<Type> map_invcov__ = map_cov__.inverse();
Type map_nll__ = Type(0.5) * atomic::logdet(map_cov__) + Type(0.5) * (pars_eps__ * (map_invcov__ * pars_eps__)).sum();
nll__ += map_nll__;
REPORT(map_nll__);
REPORT(map_pars__);
REPORT(pars_eps__);
}

//////////// Report //////////////
REPORT(Innovation);
REPORT(InnovationCovariance);
REPORT(xPrior);
REPORT(xPost);
REPORT(pPrior);
REPORT(pPost);
REPORT(nll_report);
return nll__;
}
