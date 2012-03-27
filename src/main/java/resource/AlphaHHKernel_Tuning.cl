    // OpenCL Kernel Function for Hodgkin Huxley integration step
    kernel void IntegrateHHStep(const float maxG_K,
    							const float maxG_Na,
    							const float maxG_Leak,
    							const float E_K,
    							const float E_Na,
    							const float E_Leak,
    							const float dt,
    							const int steps,
    							global float* I_ext,
    							global float* V_in, 
    							global float* x_n_in,
    							global float* x_m_in,
    							global float* x_h_in,
    							global float* V_results,
    							global float* Xn_results,
    							global float* Xm_results,
    							global float* Xh_results,
    							int numElements) {
        // get index into global data array
        int iGID = get_global_id(0);

        // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
        if (iGID >= numElements)  {
            return;
        }
        
        // here we go, HH integration loop (Euler's method)
    	for (float t = 0; t < steps; t+=1) {
           
			// logic for step integration
        	// alpha functions
			float4 alpha = (float4)((10 - V_in[iGID]) / (100 * (exp((10 - V_in[iGID]) / 10) - 1)),
									(25 - V_in[iGID]) / (10 * (exp((25 - V_in[iGID]) / 10) - 1)),
									0.07 * exp(-V_in[iGID] / 20),
									0.0f);
			// beta functions
			float4 beta = (float4)(0.125 * exp(-V_in[iGID] / 80),
								   4 * exp(-V_in[iGID] / 18),
								   1 / (exp((30 - V_in[iGID]) / 10) + 1),
								   0.0f);
	
			// calculate tau and x0 with alpha and beta
			float4 tau;
	 		tau = 1.0f / (alpha + beta);
	
			float4 x0;
			x0 = alpha * tau;
	
			// leaky integration for Xs with eurler's method
			x_n_in[iGID] = (1 - dt / tau.x) * x_n_in[iGID] + dt / tau.x * x0.x;
			x_m_in[iGID] = (1 - dt / tau.y) * x_m_in[iGID] + dt / tau.y * x0.y;
			x_h_in[iGID] = (1 - dt / tau.z) * x_h_in[iGID] + dt / tau.z * x0.z;
	
			// calculate conductances for n, m, h
			float4 gnmh = (float4)(maxG_K * pow(x_n_in[iGID], 4),
								   maxG_Na * pow(x_m_in[iGID], 3) * x_h_in[iGID],
								   maxG_Leak,
								   0.0f);
	
			// calculate current with Ohm's law
			float4 I = (float4)(gnmh.x * (V_in[iGID] - E_K),
								gnmh.y * (V_in[iGID] - E_Na),
								gnmh.z * (V_in[iGID] - E_Leak),
								0.0f);
	
			// given all the currents, update voltage membrane
			V_in[iGID] = V_in[iGID] + dt * (I_ext[iGID] - (I.x + I.y + I.z));
			
			// store results for each step
			V_results[iGID + (int)t*numElements] = V_in[iGID];
			Xn_results[iGID + (int)t*numElements] = x_n_in[iGID];
			Xm_results[iGID + (int)t*numElements] = x_m_in[iGID];
			Xh_results[iGID + (int)t*numElements] = x_h_in[iGID];
		}
    }