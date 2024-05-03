functions{

  // Transforms time into x, as given by model
  real y_model(
                  real t, 
                  real atp0,
                  real adp0,
                  real p0,
                  real ktime,  
                  real tau){ 

                  // Motor concentration 
                  real m = 1; 

                  // Keff calculation
                  // real keff = K_T * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP ) / ( (1/K_T) - (1/KD) - (1/KP) );
                  // real keff = C1*( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP );

                  // Ktime calculation
                  // real ktime = ((1/K_T) - (1/KD) - (1/KP)) * keff / ( gamma * m );
                  // real ktime = C2*( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP ); 

                  // Result
                  real result = atp0 * exp(-(t + tau)/ktime); 

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


parameters {
        // real<lower=0> C1;                  
        // real<lower=0> C2;                 
        // real<lower=0> KD;                  // in units of uM
        // real<lower=0> KP;                  // in units of uM
        real<lower=0> ktime;             // units of s
        real<lower=0> tau;             // units of s
}

model {
        // Priors for parameters
        // C1 ~ normal(100.0, 10.0);
        // C2 ~ normal(50.0, 10.0);
        // KD ~ normal(50.0, 10.0);
        ktime ~ normal(500, 100.0);
        tau ~ normal(50, 10.0);

        // real C2 = 500; 
        // real KD = 50; 
        // real ktime = 50;
        // real tau = 240;

        // Likelihood 
        for (j in 1:N) {
          atp[j] ~ normal(y_model(time[j], atp0[j], adp0[j], p0[j], ktime, tau), 5);
        }
      }

generated quantities {
        array[N] real y_theoretical_values;
        for (i in 1:N) {
          y_theoretical_values[i] = y_model(time[i], atp0[i], adp0[i], p0[i], ktime, tau);
        }
}





