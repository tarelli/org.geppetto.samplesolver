package org.openworm.simulationengine.samplesolver;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.openworm.simulationengine.core.solver.ISolver;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;


/**
 * A Spring and JUnit system test for the ISolver interface. System tests
 * like this use Spring to initialize the system implementation, then test the
 * interaction of the various components through the service interface to carry
 * out service use-cases. The Spring configuration file read to initialize the
 * system is located in this same package.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration
public class ExampleServiceSystemTests {

	@Autowired
	private ISolver service;

	@Test
	public void testDoIt() {

	}
}
