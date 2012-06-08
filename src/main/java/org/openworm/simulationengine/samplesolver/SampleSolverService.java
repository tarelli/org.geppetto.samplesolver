package org.openworm.simulationengine.samplesolver;

import static java.lang.System.nanoTime;
import static java.lang.System.out;

import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import org.bridj.Pointer;
import org.openworm.simulationengine.core.model.HHModel;
import org.openworm.simulationengine.core.model.IModel;
import org.openworm.simulationengine.core.simulation.ITimeConfiguration;
import org.openworm.simulationengine.core.solver.ISolver;
import org.springframework.stereotype.Service;

import static org.bridj.Pointer.allocateFloats;
import com.nativelibs4java.opencl.CLBuffer;
import com.nativelibs4java.opencl.CLContext;
import com.nativelibs4java.opencl.CLEvent;
import com.nativelibs4java.opencl.CLKernel;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.opencl.CLPlatform.DeviceFeature;
import com.nativelibs4java.opencl.CLProgram;
import com.nativelibs4java.opencl.CLQueue;
import com.nativelibs4java.opencl.JavaCL;
import com.nativelibs4java.util.IOUtils;

/**
 * A simple implementation of the ISolver interface for solving Hodking Huxley
 * equation given models. This implementation is internal to this module and is
 * not exported to other bundles.
 */
@Service
public class SampleSolverService implements ISolver {

	private String KERNEL_PATH = "/resource/AlphaHHKernel_Tuning.cl";
	private String KERNEL_NAME = "IntegrateHHStep";

	private List<IModel> _models;

	// max conductances
	float maxG_K = 36;
	float maxG_Na = 120;
	float maxG_Leak = (float) 0.3;
	// reverse potentials
	float E_K = -12;
	float E_Na = 115;
	float E_Leak = (float) 10.613;

	public List<List<IModel>> solve(final List<IModel> models, final ITimeConfiguration timeConfiguration) {
		out.println("Solver invoked with " + models.size() + " models");

		_models = models;
		List<List<IModel>> results = null;
		int ELEM_COUNT = models.size();

		try {
			CLContext context = JavaCL.createBestContext(DeviceFeature.CPU);
			out.println(context.getDevices()[0].toString());
			CLQueue queue = context.createDefaultQueue();
			ByteOrder byteOrder = context.getByteOrder();

			// Read the program sources and compile them :
			String src = IOUtils.readText(SampleSolverService.class.getResource(KERNEL_PATH));
			CLProgram program = context.createProgram(src);

			// I/O BUFFERS DECLARATION
			Pointer<Float> I_in_Ptr = allocateFloats(ELEM_COUNT).order(byteOrder);
			Pointer<Float> V_in_Ptr = allocateFloats(ELEM_COUNT).order(byteOrder);
			Pointer<Float> x_n_in_Ptr = allocateFloats(ELEM_COUNT).order(byteOrder);
			Pointer<Float> x_m_in_Ptr = allocateFloats(ELEM_COUNT).order(byteOrder);
			Pointer<Float> x_h_in_Ptr = allocateFloats(ELEM_COUNT).order(byteOrder);

			// fill input buffers with initial conditions
			initInputBuffers(models, I_in_Ptr, V_in_Ptr, x_n_in_Ptr, x_m_in_Ptr, x_h_in_Ptr);

			// Create OpenCL input buffers (using the native memory pointers) :
			CLBuffer<Float> I_in_Buffer = context.createFloatBuffer(Usage.Input, I_in_Ptr);
			CLBuffer<Float> V_in_Buffer = context.createFloatBuffer(Usage.Input, V_in_Ptr);
			CLBuffer<Float> x_n_in_Buffer = context.createFloatBuffer(Usage.Input, x_n_in_Ptr);
			CLBuffer<Float> x_m_in_Buffer = context.createFloatBuffer(Usage.Input, x_m_in_Ptr);
			CLBuffer<Float> x_h_in_Buffer = context.createFloatBuffer(Usage.Input, x_h_in_Ptr);
			CLBuffer<Float> V_results_Buffer = context.createFloatBuffer(Usage.Output, ELEM_COUNT * timeConfiguration.getTimeSteps());
			CLBuffer<Float> Xn_results_Buffer = context.createFloatBuffer(Usage.Output, ELEM_COUNT * timeConfiguration.getTimeSteps());
			CLBuffer<Float> Xm_results_Buffer = context.createFloatBuffer(Usage.Output, ELEM_COUNT * timeConfiguration.getTimeSteps());
			CLBuffer<Float> Xh_results_Buffer = context.createFloatBuffer(Usage.Output, ELEM_COUNT * timeConfiguration.getTimeSteps());

			// get a reference to the kernel function
			CLKernel integrateHHStepKernel = program.createKernel(KERNEL_NAME);

			long compuTime = nanoTime();

			integrateHHStepKernel.setArgs(maxG_K, maxG_Na, maxG_Leak, 
										  E_K, E_Na, E_Leak, 
										  timeConfiguration.getTimeStepLength(), timeConfiguration.getTimeSteps(), 
										  I_in_Buffer, V_in_Buffer, x_n_in_Buffer, x_m_in_Buffer, x_h_in_Buffer,
										  V_results_Buffer, Xn_results_Buffer, Xm_results_Buffer, Xh_results_Buffer, ELEM_COUNT);

			int[] globalSizes = new int[] { ELEM_COUNT };
			CLEvent integrateEvt = integrateHHStepKernel.enqueueNDRange(queue, globalSizes);

			// blocks until add_floats finished
			Pointer<Float> V_out_Ptr = V_results_Buffer.read(queue,	integrateEvt);
			Pointer<Float> x_n_out_Ptr = Xn_results_Buffer.read(queue, integrateEvt);
			Pointer<Float> x_m_out_Ptr = Xm_results_Buffer.read(queue, integrateEvt);
			Pointer<Float> x_h_out_Ptr = Xh_results_Buffer.read(queue, integrateEvt);

			compuTime = nanoTime() - compuTime;

			out.println("computation took: " + (compuTime / 1000000) + "ms");
			out.println("end of solver computation");

			// return all the models sampled as specified in timeConfiguration
			results = convertBufferToModel(V_out_Ptr, x_n_out_Ptr, x_m_out_Ptr, x_h_out_Ptr, models.size(), timeConfiguration);
		} catch (Exception e) {
			// TODO: need to handle exceptions
			e.printStackTrace();
		} finally {
			System.out.println("End of HH simulation");
		}

		return results;
	}

	/**
	 * Given float buffers with all the results generates IModels
	 * 
	 * @param vBuffer: a buffer with all the v result values for each time step
	 * @param xhBuffer: a buffer with all the xh result values for each time step
	 * @param xnBuffer: a buffer with all xn result values for each time step
	 * @param xmBuffer: a buffer with all xn result values for each time step
	 * @param noModels: total number of models being evaluated
	 * @param timeConfiguration: time configuration for this solver run
	 * @return
	 */
	private List<List<IModel>> convertBufferToModel(Pointer<Float> vBuffer,	Pointer<Float> xnBuffer, Pointer<Float> xmBuffer, Pointer<Float> xhBuffer, Integer noModels, ITimeConfiguration timeConfiguration) {
		List<List<IModel>> allModels = new ArrayList<List<IModel>>();
		int currentModel = 1;
		int timeStep = 1;

		for (int i = 0; i < noModels * timeConfiguration.getTimeSteps(); i++) {
			// current model is not contained yet, we have to add it
			if (currentModel > allModels.size()) {
				allModels.add(new ArrayList<IModel>());
			}

			// sample when the time step divided by the sample is not decimal
			// NOTE: else what?
			if (((float) timeStep / timeConfiguration.getSamplePeriod()) % 1 == 0) {
				// sample result
				allModels.get(currentModel - 1).add(new HHModel(_models.get(currentModel - 1).getId(), vBuffer.get(i), xnBuffer.get(i), xmBuffer.get(i), xhBuffer.get(i), 0.0f));
			}

			// advance or reset currentModel
			if (currentModel == noModels) {
				currentModel = 1;
				timeStep++;
			} else {
				// advance
				currentModel++;
			}
		}

		return allModels;
	}

	/**
	 * Input buffer initialization given a list of models with initial
	 * conditions for the current run
	 * 
	 * @param models: a list of models containing initial conditions
	 * @param V_in: input buffer with V initial conditions
	 * @param x_n_in: input buffer with xh initial conditions
	 * @param x_m_in: input buffer with xm initial conditions
	 * @param x_h_in: input buffer with xh initial conditions
	 */
	private void initInputBuffers(List<IModel> models, Pointer<Float> i_in, Pointer<Float> V_in, Pointer<Float> x_n_in, Pointer<Float> x_m_in, Pointer<Float> x_h_in) {
		// load input buffers from models
		for (int y = 0; y < models.size(); y++) {
			HHModel model = (HHModel) models.get(y);
			V_in.set(y, model.getV());
			x_n_in.set(y, model.getXn());
			x_m_in.set(y, model.getXm());
			x_h_in.set(y, model.getXh());
			i_in.set(y, model.getI());
		}
	}
};