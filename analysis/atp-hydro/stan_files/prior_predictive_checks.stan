functions{

  // Transforms time into x, as given by model
  real x_variable(
                  real t, 
                  real atp0,
                  real adp0,
                  real p0,
                  real K_T, 
                  real K_D, 
                  real K_P, 
                  real gamma, 
                  real tau){ 

                  // Motor concentration 
                  real m = 1; 

                  // Keff calculation
                  real keff = K_T * ( 1 + ( atp0 + adp0 ) / K_D + ( atp0 + p0 ) / K_P ) / ( (1/K_T) - (1/K_D) - (1/K_P) );

                  // Ktime calculation
                  real ktime = ((1/K_T) - (1/K_D) - (1/K_P)) * keff / ( gamma * m );

                  // Result
                  real result = exp(atp0/keff + log(atp0/keff) - (t + tau)/ktime);

                  return result <= 0 ? 0.00001 : result;
                  
                  }


  // Theoretical ATP concentration using Lambert function
  real y_model(
                  real t, 
                  real atp0,
                  real adp0,
                  real p0,
                  real K_T, 
                  real K_D, 
                  real K_P, 
                  real gamma, 
                  real tau){ 

                  // Keff calculation
                  real keff = K_T * ( 1 + ( atp0 + adp0 ) / K_D + ( atp0 + p0 ) / K_P ) / ( (1/K_T) - (1/K_D) - (1/K_P) );

                  // Transform time into x
                  real x = x_variable(t, 
                                      atp0,
                                      adp0,
                                      p0,
                                      K_T, 
                                      K_D, 
                                      K_P, 
                                      gamma, 
                                      tau);

                  // Result
                  real result = keff*lambert_w0(x);

                  return result <= 0 ? 0.00001 : result;
                  
                  }
    
}

data {
        int N;                     // Total number of data points over entire data set (number of curves * number of data points per curve)
        array[N] real atp; 
        array[N] real time; 
        array[N] real atp0;
        array[N] real adp0;
        array[N] real p0;
}


generated quantities {
  
  // Draw Parameters
  real<lower=0> KT;                  // KT in units of uM
  real<lower=0> KD;                  // KD in units of uM
  real<lower=0> KP;                  // KP in units of uM
  real<lower=0> gamma;               // units of 1/s
  real<lower=0> tau;                 // units of s

  KT = normal_rng(50.0, 10.0);
  KD = normal_rng(50.0, 10.0);
  KP = normal_rng(100.0, 50.0);
  tau = normal_rng(2.5, 0.1);
  gamma = normal_rng(1.0, 0.1);

  // Calculate Theoretical atp conc and Likelihood
  array[N] real y_theory;
  array[N] real l;

  for (i in 1:N) {
    y_theory[i] = y_model(time[i], atp0[i], adp0[i], p0[i], KT, KD, KP, gamma, tau);
    l[i] = normal_rng(y_theory[i], 0.1);
  }
}








