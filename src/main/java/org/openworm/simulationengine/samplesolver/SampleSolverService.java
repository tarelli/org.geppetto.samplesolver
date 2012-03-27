package org.openworm.simulationengine.samplesolver;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;
import static com.jogamp.opencl.CLMemory.Mem.WRITE_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static java.lang.System.nanoTime;
import static java.lang.System.out;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import org.openworm.simulationengine.core.model.HHModel;
import org.openworm.simulationengine.core.model.IModel;
import org.openworm.simulationengine.core.simulation.ITimeConfiguration;
import org.openworm.simulationengine.core.solver.ISolver;
import org.springframework.stereotype.Service;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

/**
 * A simple implementation of the ISolver interface for solving Hodking Huxley equation given models. 
 * This implementation is internal to this module and is not exported to other bundles.
 */
@Service
public class SampleSolverService implements ISolver {
	
	private String KERNEL_PATH = "/resource/AlphaHHKernel_Tuning.cl";
	private String KERNEL_NAME = "IntegrateHHStep";
	
	private List<IModel> _models;
	
	
	public List<List<IModel>> solve(final List<IModel> models,final ITimeConfiguration timeConfiguration)
	{
		out.println("Solver invoked with " + models.size() + " models");

		_models=models;
		
		// set up (uses default CLPlatform and creates context for all devices)
		CLContext context = CLContext.create();
		out.println("created "+ context);

		try{
			// an array with available devices
			CLDevice[] devices = context.getDevices();

			for(int i=0; i<devices.length; i++)
			{
				out.println("device-" + i + ": " + devices[i]);
			}	

			// have a look at the output and select a device ...
			CLDevice device = devices[0];
			// ... or use this code to select "fastest" device
			//CLDevice device = context.getMaxFlopsDevice();

			out.println("using "+ device);

			// create a command queue on the selected device.
			CLCommandQueue queue = device.createCommandQueue();

			/* CONSTANTS */
			// TODO: all this stuff should be moved to the model (but it's OK for this alpha version!)
			// max conductances
			float maxG_K = 36;
			float maxG_Na = 120;
			float maxG_Leak = (float) 0.3;
			// reverse potentials 
			float E_K = -12;
			float E_Na = 115;
			float E_Leak = (float) 10.613;
			/* CONSTANTS */

			// Length of arrays to process
			int elementCount = models.size();
			// Local work size dimensions for the selected device
			int localWorkSize = 0;//min(device.getMaxWorkGroupSize(), 256);
			// rounded up to the nearest multiple of the localWorkSize
			int globalWorkSize = elementCount;
			// results buffers are bigger because we are capturing every value for every item for every time-step
			int globalWorkSize_Results = elementCount*timeConfiguration.getTimeSteps();

			// load sources, create and build program
			CLProgram program = null;
			try {
				program = context.createProgram(SampleSolverService.class.getResourceAsStream(KERNEL_PATH)).build();
			} catch (IOException e) {
				out.println("Something went *horribly* wrong when loading the kernel!");
			}

			/* I/O BUFFERS DECLARATION */
			CLBuffer<FloatBuffer> I_in_Buffer = context.createFloatBuffer(globalWorkSize, READ_ONLY);
			CLBuffer<FloatBuffer> V_in_Buffer = context.createFloatBuffer(globalWorkSize, READ_WRITE);
			CLBuffer<FloatBuffer> x_n_in_Buffer = context.createFloatBuffer(globalWorkSize, READ_WRITE);
			CLBuffer<FloatBuffer> x_m_in_Buffer = context.createFloatBuffer(globalWorkSize, READ_WRITE);
			CLBuffer<FloatBuffer> x_h_in_Buffer = context.createFloatBuffer(globalWorkSize, READ_WRITE);
			CLBuffer<FloatBuffer> V_results_Buffer = context.createFloatBuffer(globalWorkSize_Results, WRITE_ONLY);
			CLBuffer<FloatBuffer> Xn_results_Buffer = context.createFloatBuffer(globalWorkSize_Results, WRITE_ONLY);
			CLBuffer<FloatBuffer> Xm_results_Buffer = context.createFloatBuffer(globalWorkSize_Results, WRITE_ONLY);
			CLBuffer<FloatBuffer> Xh_results_Buffer = context.createFloatBuffer(globalWorkSize_Results, WRITE_ONLY);
			/* I/O BUFFERS DECLARATION */
			
			// calculates memory used by the device for the input buffers (we have 5 of them, same size)
			out.println("Approx. used device memory (input buffers only): " + (V_in_Buffer.getCLSize()*5)/1000000 +"MB");

			// fill input buffers with initial conditions 
			initInputBuffers(models, V_in_Buffer.getBuffer(), x_n_in_Buffer.getBuffer(), x_m_in_Buffer.getBuffer(), x_h_in_Buffer.getBuffer(), I_in_Buffer.getBuffer());
			// initialize results buffers
			initResultsBuffers(V_results_Buffer.getBuffer(), Xn_results_Buffer.getBuffer(), Xm_results_Buffer.getBuffer(),Xh_results_Buffer.getBuffer());

			// get a reference to the kernel function with the name 'IntegrateHHStep'
			CLKernel kernel = program.createCLKernel(KERNEL_NAME);

			long compuTime = nanoTime();

			// map the input/output buffers to its input parameters in the kernel
			kernel.putArg(maxG_K)
			.putArg(maxG_Na)
			.putArg(maxG_Leak)
			.putArg(E_K)
			.putArg(E_Na)
			.putArg(E_Leak)
			.putArg(timeConfiguration.getTimeStepLength())
			.putArg(timeConfiguration.getTimeSteps())
			.putArg(I_in_Buffer)
			.putArgs(V_in_Buffer, x_n_in_Buffer, x_m_in_Buffer, x_h_in_Buffer)
			.putArgs(V_results_Buffer, Xn_results_Buffer, Xm_results_Buffer, Xh_results_Buffer)
			.putArg(elementCount)
			.rewind();

			// asynchronous write of data to GPU device, followed by blocking read to get the computed results back
			queue
			.putWriteBuffer(I_in_Buffer, false)
			.putWriteBuffer(V_in_Buffer, false)
			.putWriteBuffer(x_n_in_Buffer, false)
			.putWriteBuffer(x_m_in_Buffer, false)
			.putWriteBuffer(x_h_in_Buffer, false)
			.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
			// true = blocking output buffers, this causes the data to get sent back (we need to get them back)
			// NOTE: this is the performance bottleneck!
			.putReadBuffer(V_results_Buffer, true)
			.putReadBuffer(Xn_results_Buffer, true)
			.putReadBuffer(Xm_results_Buffer, true)
			.putReadBuffer(Xh_results_Buffer, true);

			compuTime = nanoTime() - compuTime;

			out.println("computation took: "+ (compuTime/1000000) +"ms");
			out.println("end of solver computation");

			// return all the models sampled as specified in timeConfiguration
			return convertBufferToModel(V_results_Buffer.getBuffer(), Xn_results_Buffer.getBuffer(), Xm_results_Buffer.getBuffer(), Xh_results_Buffer.getBuffer(), models.size(), timeConfiguration);
		}
		finally
		{
			// cleanup all resources associated with this context.
			context.release();
		}		
	}


	/**
	 * Given float buffers with all the results generates IModels
	 * 
	 * @param vBuffer : a buffer with all the v result values for each time step
	 * @param xhBuffer : a buffer with all the xh result values for each time step
	 * @param xnBuffer : a buffer with all xn result values for each time step
	 * @param xmBuffer : a buffer with all xn result values for each time step
	 * @param noModels : total number of models being evaluated
	 * @param timeConfiguration : time configuration for this solver run 
	 * @return
	 */
	private List<List<IModel>> convertBufferToModel(FloatBuffer vBuffer, FloatBuffer xnBuffer, FloatBuffer xmBuffer, FloatBuffer xhBuffer, Integer noModels, ITimeConfiguration timeConfiguration) 
	{
		List<List<IModel>> allModels=new ArrayList<List<IModel>>();
		int currentModel=1;
		int timeStep=1;	
	
		for(int i=0;i<noModels*timeConfiguration.getTimeSteps();i++)
		{
			// current model is not contained yet, we have to add it
			if(currentModel>allModels.size())
			{
				allModels.add(new ArrayList<IModel>());
			}
			
			// sample when the time step divided by the sample is not decimal
			// NOTE: else what?
			if(  ((float)timeStep/timeConfiguration.getSamplePeriod())%1==0)  
			{
				// sample result
				allModels.get(currentModel-1).add(new HHModel(_models.get(currentModel-1).getId(),vBuffer.get(i), xnBuffer.get(i), xmBuffer.get(i), xhBuffer.get(i),0.0f));			
			}
			
			// advance or reset currentModel
			if(currentModel==noModels)
			{
				currentModel=1;
				timeStep++;
			}
			else
			{
				//advance
				currentModel++;
			}
		}

		return allModels;
	}


	/**
	 * Input buffer initialization given a list of models with initial conditions for the current run
	 * 
	 * @param models : a list of models containing initial conditions
	 * @param V_in : input buffer with V initial conditions
	 * @param x_n_in : input buffer with xh initial conditions
	 * @param x_m_in: input buffer with xm initial conditions
	 * @param x_h_in: input buffer with xh initial conditions
	 */
	private void initInputBuffers(List<IModel> models, FloatBuffer V_in, FloatBuffer x_n_in, FloatBuffer x_m_in, FloatBuffer x_h_in, FloatBuffer i_in) 
	{
		// load input buffers from models
		for(int y=0; y<models.size(); y++)
		{
			HHModel model = (HHModel)models.get(y);
			V_in.put(model.getV());
			x_n_in.put(model.getXn());
			x_m_in.put(model.getXm());
			x_h_in.put(model.getXh());
			i_in.put(model.getI());
		}

		V_in.rewind();
		x_n_in.rewind();      
		x_m_in.rewind();
		x_h_in.rewind();
		i_in.rewind();
	}

	/**
	 * Inits all the output buffers with a bunch of float z(h)eroes.
	 * 
	 * @param vBuffer
	 * @param xhBuffer 
	 * @param xnBuffer 
	 * @param xmBuffer 
	 */
	private void initResultsBuffers(FloatBuffer vBuffer, FloatBuffer xnBuffer, FloatBuffer xmBuffer, FloatBuffer xhBuffer) 
	{
		initBuffer(vBuffer, 0.0f);
		initBuffer(xnBuffer, 0.0f);
		initBuffer(xmBuffer, 0.0f);
		initBuffer(xhBuffer, 0.0f);
	}


	/**
	 * Inits a a given float buffer with a given value
	 * 
	 * @param vBuffer
	 */
	private void initBuffer(FloatBuffer buffer, Float initValue) 
	{
		while(buffer.remaining() != 0)
		{
			buffer.put(initValue);
		}
		buffer.rewind();
	}

	/**
	 * Rounds up to the nearest multiple of the groupsize
	 * 
	 * @param groupSize
	 * @param globalSize
	 * @return
	 */
	private int roundUp(int groupSize, int globalSize) 
	{
		int r = globalSize % groupSize;
		if (r == 0) {
			return globalSize;
		} else {
			return globalSize + groupSize - r;
		}
	}
}
;