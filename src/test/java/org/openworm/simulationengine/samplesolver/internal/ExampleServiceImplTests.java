package org.openworm.simulationengine.samplesolver.internal;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.junit.Test;
import org.openworm.simulationengine.core.model.HHModel;
import org.openworm.simulationengine.core.model.IModel;
import org.openworm.simulationengine.core.simulation.ITimeConfiguration;
import org.openworm.simulationengine.core.simulation.TimeConfiguration;
import org.openworm.simulationengine.samplesolver.SampleSolverService;

/**
 * JUnit test for the example solver implementation. Such a unit test tests
 * internal use-case coordination logic in isolation of other dependencies.
 * Spring is not used in an unit test environment.
 */
public class ExampleServiceImplTests {
	public static Random randomGenerator = new Random();
	private SampleSolverService alphaSolver = new SampleSolverService();

	/**
	 * Run the solver in one single go and generates some output plots for visual reference.
	 * NOTE: it passes if no errors are thrown
	 */
	@Test
	public void testSolveInSingleGo() {
		// define some parameters for the test
		boolean PLOTTING = false;
		int SAMPLES = 3;
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		float I_Ext = 0.0f;
		
		// create the 302 models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
			
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j), -10, 0, 0, 1, I_Ext));
		}
		
		// invoke solve method
		ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps,1);
		List<List<IModel>> resultsBuffer = alphaSolver.solve(models, timeConfig);
		
		if(PLOTTING)
		{
			// some dictionary for plotting
			Hashtable<Integer, Hashtable<Float, Float>> V_by_t = new Hashtable<Integer, Hashtable<Float, Float>>();
			List<Integer> sampleIndexes = new ArrayList<Integer>();
			// Generate some random indexes in the 0 .. ELEM_COUNT range
			for(int i = 0; i < SAMPLES; i++ )
			{
				sampleIndexes.add(randomGenerator.nextInt(ELEM_COUNT));
			}
			
	    	// record some values for plotting
	    	Iterator<Integer> itr = sampleIndexes.iterator();
	    	while(itr.hasNext())
	    	{   
	    		Integer index = itr.next();
	
	    		if(!V_by_t.containsKey(index))
	    		{
	    			V_by_t.put(index, new Hashtable<Float, Float>());
	    		}
	    		
	    		for(int j = 0; j < steps/timeConfig.getSamplePeriod(); j++)
	    		{
	    			V_by_t.get(index).put(new Float(j*dt + START_TIME), ((HHModel)resultsBuffer.get(index).get(j)).getV());
	    		}
	    	}
	    	
	    	// print some sampled charts to make sure we got fine-looking results.
	    	// Plot results
	    	Iterator<Integer> iter = sampleIndexes.iterator();
	    	while(iter.hasNext())
	    	{   
	    		XYSeries series = new XYSeries("HH_Graph");
	
	    		Integer index = iter.next();
	
	    		for (int t = 0; t < steps/timeConfig.getSamplePeriod(); t++) {
	    			series.add(t*dt + START_TIME, V_by_t.get(index).get(t*dt + START_TIME));
	    		}
	
	    		// Add the series to your data set
	    		XYSeriesCollection dataset = new XYSeriesCollection();
	    		dataset.addSeries(series);
	
	    		plot(dataset, index);
	    	}
		}
	}
	
	/**
	 * Run the solver in multiple steps and generates some output plots from aggregation of results.
	 * NOTE: it passes if no errors are thrown
	 */
	@Test
	public void testSolveInMultipleSteps() {
		// define some parameters for the test
		boolean PLOTTING = false;
		int SAMPLES = 3;
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		float I_Ext = 0.0f;
		
		// create the 302 models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
			
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j),-10, 0, 0, 1, I_Ext));
		}
		
		// break it in 100 intervals and append results each time the solver runs
		int SCALE_FACTOR = 100;
		ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps/SCALE_FACTOR,1);
		List<List<IModel>> globalResultsBuffer = new ArrayList<List<IModel>> ();
		List<List<IModel>> tempResultsBuffer = new ArrayList<List<IModel>> ();
		for(int c = 0; c < SCALE_FACTOR; c++)
		{	
			if(c > 0)
			{
				// clear models to pass down to solver
				models.clear();
				// iterate through results of latest run
				Iterator itr = tempResultsBuffer.iterator(); 
				while(itr.hasNext()) {
				    List<IModel> modelSnapshots = (List<IModel>) itr.next(); 
				    ((HHModel)modelSnapshots.get(steps/SCALE_FACTOR-1)).setI(I_Ext);
					// grab final conditions from previous cycle
					models.add(modelSnapshots.get(steps/SCALE_FACTOR-1));
				} 
			}
			
			// invoke solve method multiple times and put together all the results
			tempResultsBuffer = alphaSolver.solve(models, timeConfig);
			
			// add results to global results buffer
			if(globalResultsBuffer.size() == 0)
		    {
				// if it's the first time add them all
				globalResultsBuffer.addAll(tempResultsBuffer);
		    }
			else
			{
				// iterate through latest results
				for (int r = 0; r < tempResultsBuffer.size(); r++)
				{
					// add all the snapshots relative to the latest solver run for each of the elements
					globalResultsBuffer.get(r).addAll(tempResultsBuffer.get(r));
				}
			}
		}
		
		if(PLOTTING)
		{
			// some dictionary for plotting
			Hashtable<Integer, Hashtable<Float, Float>> V_by_t = new Hashtable<Integer, Hashtable<Float, Float>>();
			List<Integer> sampleIndexes = new ArrayList<Integer>();
			// Generate some random indexes in the 0 .. ELEM_COUNT range
			for(int i = 0; i < SAMPLES; i++ )
			{
				sampleIndexes.add(randomGenerator.nextInt(ELEM_COUNT));
			}
			
	    	// record some values for plotting
	    	Iterator<Integer> itr = sampleIndexes.iterator();
	    	while(itr.hasNext())
	    	{   
	    		Integer index = itr.next();
	
	    		if(!V_by_t.containsKey(index))
	    		{
	    			V_by_t.put(index, new Hashtable<Float, Float>());
	    		}
	    		
	    		for(int j = 0; j < steps/timeConfig.getSamplePeriod(); j++)
	    		{
	    			V_by_t.get(index).put(new Float(j*dt + START_TIME), ((HHModel)globalResultsBuffer.get(index).get(j)).getV());
	    		}
	    	}
	    	
	    	// print some sampled charts to make sure we got fine-looking results.
	    	// Plot results
	    	Iterator<Integer> iter = sampleIndexes.iterator();
	    	while(iter.hasNext())
	    	{   
	    		XYSeries series = new XYSeries("HH_Graph");
	
	    		Integer index = iter.next();
	
	    		for (int t = 0; t < steps/timeConfig.getSamplePeriod(); t++) {
	    			series.add(t*dt + START_TIME, V_by_t.get(index).get(t*dt + START_TIME));
	    		}
	
	    		// Add the series to your data set
	    		XYSeriesCollection dataset = new XYSeriesCollection();
	    		dataset.addSeries(series);
	
	    		plot(dataset, index);
	    	}
		}
	}
	
	/**
	 * Tests the solver in the scenario of I_Ext = 0 over 130 ms in one single go.
	 * NOTE: reference curve is "I_ext 0" to be found in reference_curves folder in this project
	 */
	@Test
	public void testSolverInSingleGoWithNoExtCurrent() {
		// define some parameters for the test
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		// calculate how many steps for the given dt
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		
		// create the models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j), -10, 0, 0, 1,/*set current to 0 and never change it*/0));
		}
		
		// invoke solve method
		ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps,1);
		List<List<IModel>> resultsBuffer = alphaSolver.solve(models, timeConfig);
		
		for (int j = 0; j < steps/timeConfig.getSamplePeriod(); j++)
		{
			for (int c = 0; c < ELEM_COUNT; c++)
			{
				//check that curve looks ok - in this case ("I_ext 0.png")
				
				// get V
				float V = ((HHModel)resultsBuffer.get(c).get(j)).getV();
				// get translated time (to match coordinates in the reference figs)
				float translatedTime = j*dt + START_TIME;
				
				String message =  "t=" + translatedTime + ";" + "V=" + V;
				
				if(translatedTime > -30 && translatedTime < -28.5)
				{
					// in the negative (but never lower than -10)
					assertTrue(message, V > -10 && V < 0);
				}
				else if(translatedTime > -28 && translatedTime < -26)
				{
					// going up but below 100
					assertTrue(message, V > 0 && V < 110);
				}
				else if(translatedTime > -26 && translatedTime < -25.5)
				{
					// spiking here
					assertTrue(message, V > 75 && V < 115);
				}
				else if(translatedTime > -25 && translatedTime < -24.5)
				{
					// coming back down
					assertTrue(message, V > 60 && V < 90);
				}
				else if(translatedTime > -24.5 && translatedTime < -23.5)
				{
					// coming down but still positive
					assertTrue(message, V > 0 && V < 65);
				}
				else if(translatedTime > -23 && translatedTime < -22)
				{	
					// over-polarization here
					assertTrue(message, V > -15 && V < 0);
				}
				else if(translatedTime > -7 && translatedTime < -5)
				{	
					// comes back above 0
					assertTrue(message, V > 0.25);
				}
				else if(translatedTime > 0)
				{
					// stable around 0
					assertTrue(message, V > -0.1 && V < 0.1);
				}
			}
		}
	}
	
	/**
	 * Tests the solver in the scenario of I_Ext = 0 over 130 ms in one single go.
	 * NOTE: reference curve is "I_ext 0" to be found in reference_curves folder in this project
	 */
	@Test
	public void testSolverInMultipleStepsWithNoExtCurrent() {
		// define some parameters for the test
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		// calculate how many steps for the given dt
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		
		// create the models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j), -10, 0, 0, 1,/*set current to 0 and never change it*/0));
		}
		
		// break it in 100 intervals and append results each time the solver runs
		int SCALE_FACTOR = 100;
		ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps/SCALE_FACTOR,1);
		List<List<IModel>> globalResultsBuffer = new ArrayList<List<IModel>> ();
		List<List<IModel>> tempResultsBuffer = new ArrayList<List<IModel>> ();
		for(int c = 0; c < SCALE_FACTOR; c++)
		{	
			if(c > 0)
			{
				// clear models to pass down to solver
				models.clear();
				// iterate through results of latest run
				Iterator itr = tempResultsBuffer.iterator(); 
				while(itr.hasNext()) {
				    List<IModel> modelSnapshots = (List<IModel>) itr.next(); 
					// grab final conditions from previous cycle
					models.add(modelSnapshots.get(modelSnapshots.size()-1));
				} 
			}
			
			// invoke solve method multiple times and put together all the results
			tempResultsBuffer = alphaSolver.solve(models, timeConfig);
			
			// add results to global results buffer
			if(globalResultsBuffer.size() == 0)
		    {
				// if it's the first time add them all
				globalResultsBuffer.addAll(tempResultsBuffer);
		    }
			else
			{
				// iterate through latest results
				for (int r = 0; r < tempResultsBuffer.size(); r++)
				{
					// add all the snapshots relative to the latest solver run for each of the elements
					globalResultsBuffer.get(r).addAll(tempResultsBuffer.get(r));
				}
			}
		}
		
		// same test criteria apply 
		for (int j = 0; j < steps/timeConfig.getSamplePeriod(); j++)
		{
			for (int c = 0; c < ELEM_COUNT; c++)
			{
				//check that curve looks ok - in this case ("I_ext 0.png") test that:
				//0. we start from negative V=-10 for t=-30
				//1. we have a spike around t=-25
				//2. shoots down to around t=-20
				//3. approaches 0 around t=-10
				//4. never "goes up" again
				
				// get V
				float V = ((HHModel)globalResultsBuffer.get(c).get(j)).getV();
				// get translated time (to match coordinates in the reference figs)
				float translatedTime = j*dt + START_TIME;
				
				String message =  "t=" + translatedTime + ";" + "V=" + V;
				
				if(translatedTime > -30 && translatedTime < -28.5)
				{
					// in the negative (but never lower than -10)
					assertTrue(message, V > -10 && V < 0);
				}
				else if(translatedTime > -28 && translatedTime < -26)
				{
					// going up but below 100
					assertTrue(message, V > 0 && V < 110);
				}
				else if(translatedTime > -26 && translatedTime < -25.5)
				{
					// spiking here
					assertTrue(message, V > 75 && V < 115);
				}
				else if(translatedTime > -25 && translatedTime < -24.5)
				{
					// coming back down
					assertTrue(message, V > 60 && V < 90);
				}
				else if(translatedTime > -24.5 && translatedTime < -23.5)
				{
					// coming down but still positive
					assertTrue(message, V > 0 && V < 65);
				}
				else if(translatedTime > -23 && translatedTime < -22)
				{	
					// over-polarization here
					assertTrue(message, V > -15 && V < 0);
				}
				else if(translatedTime > -7 && translatedTime < -5)
				{	
					// comes back above 0
					assertTrue(message, V > 0.25);
				}
				else if(translatedTime > 0)
				{
					// stable around 0
					assertTrue(message, V > -0.1 && V < 0.1);
				}
			}
		}
	}
	
	/**
	 * Tests the solver in the scenario of I_Ext = 0 over 130 ms in one single go.
	 */
	@Test
	public void testThatOutputMatchesInput() {
		// define some parameters for the test
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		// calculate how many steps for the given dt
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		
		// create the models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j),/*initial V condition is -10*/-10, 0, 0, -1,/*set current to 0 and never change it*/0));
		}
		
		// invoke solve method
		ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps,1);
		List<List<IModel>> resultsBuffer = alphaSolver.solve(models, timeConfig);
		
		assertTrue(resultsBuffer.size() == ELEM_COUNT);
		assertTrue(resultsBuffer.get(0).size() == steps);
	}
	
	/**
	 * Tests custom smapling for the the solver in the scenario of I_Ext = 0 over 130 ms in one single go
	 */
	@Test
	public void testThatOutputMatchesInputWithCustomSampling() {
		// define some parameters for the test
		int ELEM_COUNT = 30;
		// all times in ms
		float START_TIME = -30;
		float END_TIME = 100;
		float dt = (float) 0.01;
		// calculate how many steps for the given dt
		int steps = (int) ((int)(END_TIME - START_TIME)/dt);
		
		// create the models to be simulated
		List<IModel> models = new ArrayList<IModel>();	
		for(int j=0; j < ELEM_COUNT; j++)
		{
			models.add(new HHModel(Integer.toString(j),/*initial V condition is -10*/-10, 0, 0, -1,/*set current to 0 and never change it*/0));
		}
		
		for(int i = 2; i < 5; i++)
		{
			int SAMPLE_PERIOD = i;
			
			// invoke solve method
			ITimeConfiguration timeConfig=new TimeConfiguration(new Float(0.01),steps,SAMPLE_PERIOD);
			List<List<IModel>> resultsBuffer = alphaSolver.solve(models, timeConfig);
			
			assertTrue(resultsBuffer.size() == ELEM_COUNT);
			assertTrue(resultsBuffer.get(0).size() == steps / SAMPLE_PERIOD);
		}
	}
	
	
	/**
	 * Helper method for plotting
	 * 
	 * @param dataset
	 * @param index
	 */
	private void plot(XYSeriesCollection dataset, int index)
    {
    	// Generate the graph
		JFreeChart chart = ChartFactory.createXYLineChart("HH Chart", "time", "Voltage", dataset, PlotOrientation.VERTICAL, true, true, false);
		try {
			ChartUtilities.saveChartAsJPEG(new File("test_output/HH_Chart_" + index + ".jpg"), chart, 500, 300);
		} catch (IOException e) {
			System.err.println("Problem occurred creating chart.");
		}
    }
}
