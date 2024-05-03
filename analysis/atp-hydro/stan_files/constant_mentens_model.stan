// Here, we infer constants keff and ktime for each condition separately

functions {

  // Used in likelihood
  real log_y_by_y0(
    real time,      // time probe data
    real ytau,      // initial atp concentration
    real ktime, 
    real keff,
    real tau) {

    // Theoretical result converting time to log(y/ytau) + y/keff 
    real result = (tau - time)/ktime + ytau/keff;

    return result; 
  }

  vector parameter_calculation(
    real ytau, 
    real atp0,      // initial atp concentration
    real adp0,      // initial adp concentration
    real p0,        // initial phosphate concentration 
    real C1,       
    real KD,
    real KP,
    real tau) {

    // Theoretical result converting time to ATP
    real m = 1; // motor concentration explicitly coded here since it's constant. Can make this varied if data for multiple motor concentrations is collected.
    real keff = C1 * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP );
    real ktime = tau/(log(atp0/ytau) + (atp0 - ytau)/keff); 
    
    return [keff, ktime]'; 
  }

}

data {
  // int N; // total number of data points
  // array[N] real atp; 
  // array[N] real log_atp_nd;
  // array[N] real time; 
  // array[N] real atp0;
  // array[N] real adp0;
  // array[N] real p0;

  int N; // number of timepoints for each condition
  array[N] real time; // time
  array[N] real atp; 

  int M; // total number of conditions

  array[M] int end_indices; 
  array[M] real ytau;
  array[M] real atp0;
  array[M] real adp0;
  array[M] real p0;
}



parameters {
  array[M] real tau;                 // time delay to microscope
  array[M] real<lower=0> keff;                  
  array[M] real<lower=0> ktime;                  // KD in units of uM
  array[M] real<lower=0> sigma;              
  }


model {

  for (i in 1:M) {

    // Priors for parameters
    tau[i] ~ normal(5.0, 2.0);
    keff[i] ~ lognormal(5.0, 0.5);
    ktime[i] ~ lognormal(5.0, 0.5);
    sigma[i] ~ normal(0.01, 0.05); 

    // ------------------ Declare variables ------------------
    int length; // to store length of sliced atp and time data.

    if (i == 1){
        length = end_indices[1] + 1; 
    } else {
        length = end_indices[i] - end_indices[i-1]; 
    }
    array[length] real atp_slice; 
    array[length] real time_slice; 


    // ------------------ Assign to variables ------------------

    if (i == 1){
        atp_slice = atp[:end_indices[1] + 1];
        time_slice = time[:end_indices[1] + 1];
    } else {
        atp_slice = atp[end_indices[i-1] + 1:end_indices[i]];
        time_slice = time[end_indices[i-1] + 1:end_indices[i]];
    }


    // ------------------ Calculate likelihood ------------------
    for (j in 1:length){
      real transformed_atp = log(atp_slice[j]/ytau[i]) + atp_slice[j]/keff[i]; 
      transformed_atp ~ normal(log_y_by_y0(time[j], ytau[i], ktime[i], keff[i], tau[i]), sigma[i]);
    }
  }
}

generated quantities {
  // Prior predictive check
  array[N] real log_y_generated;
  array[N] real transformed_data;

  for (i in 1:M) {

    // ------------------ Declare variables ------------------
    int length; // to store length of sliced atp and time data.

    if (i == 1){
        length = end_indices[1] + 1; 
    } else {
        length = end_indices[i] - end_indices[i-1]; 
    }
    array[length] real atp_slice; 
    array[length] real time_slice; 


    // ------------------ Assign to variables ------------------

    if (i == 1){
        atp_slice = atp[:end_indices[1] + 1];
        time_slice = time[:end_indices[1] + 1];
    } else {
        atp_slice = atp[end_indices[i-1] + 1:end_indices[i]];
        time_slice = time[end_indices[i-1] + 1:end_indices[i]];
    }


    // ------------------ Generate posterior ------------------
    for (j in 1:length){
      if (i == 1){
        log_y_generated[j] = normal_rng(log_y_by_y0(time[j], ytau[i], ktime[i], keff[i], tau[i]), sigma[i]);
        transformed_data[j] = log(atp_slice[j]/ytau[i]) + atp_slice[j]/keff[i];
      } else {
        log_y_generated[end_indices[i-1] + 1 + j -1] = normal_rng(log_y_by_y0(time[j], ytau[i], ktime[i], keff[i], tau[i]), sigma[i]);
        transformed_data[end_indices[i-1] + 1 + j -1] = log(atp_slice[j]/ytau[i]) + atp_slice[j]/keff[i];
      }
    }
  }
}