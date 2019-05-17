import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import time
import Bell_EBM as ebm


def CreateBaseline(star, planet, temporal=5000, spacial=32,orbit=2):
    """
    Runs a high resoltuion version of the model.
    
    Args:
        star (ebm.Star): The star to runs the test on
        planet (ebm.Planet): The planet to run the test on
        temporal (int): The temporal resolution to run the model
        spacial (int): The spacial resolution to run the model
        orbit (int): The amount of orbits to run for the model
        
    Return:
        ndarray: Baseline time array
        ndarray: Baseline map array
        ndarray: Baseline lightcurve array
    """
    _star = star
    _planet = planet
    _planet.map = ebm.Map.Map(nlat=spacial)
    _system = ebm.System(_star, _planet)
    
    Teq = _system.get_teq()
    T0 = np.ones_like(_system.planet.map.values)*Teq
    t0 = 0.
    t1 = t0+_system.planet.Porb*orbit
    dt = _system.planet.Porb/temporal
    baselineTimes, baselineMaps = _system.run_model(T0, t0, t1, dt, verbose=False, intermediates=False)
    if (planet.orbit.e != 0.):
        T0 = baselineMaps[-1]
        t0 = baselineTimes[-1]
        t1 = t0+_system.planet.Porb
        dt = (_system.planet.Porb)/1000.
        baselineTimes, baselineMaps = _system.run_model(T0, t0, t1, dt, verbose=False, intermediates=True)
        
        baselineLightcurve = _system.lightcurve(baselineTimes, baselineMaps, bolo=False, wav=4.5e-6)
    else: 
        baselineLightcurve = _system.lightcurve(bolo=False, wav=4.5e-6)
    
    return baselineTimes, baselineMaps, baselineLightcurve


def RunTests(star, planet, points, base, basetimes, loops=1):
    """
    Runs several test of a system and returns time 
    to compute and error as comapared to baseline for each test.
    
    Args:
        star (ebm.Star): The star to runs the tests on
        planet (ebm.Planet): The planet to run the tests on
        points (2darray (n by 2)): The array of points to be tested by the model, 
            each point must contain [temporal, spacial], n points are provided
        base (ndarray): Baseline lightcurve as generated by the CreateBaseline function
        basetime (ndarray): Baseline times as generated by CreateBaseline function
        loops (int): Number of times the test will be run to avergae out the time
        
    Return:
        ndarray: Array of all tested lightcurves
        ndarray: (n by 4), n points of format [temporal, spacial, time_to_compute, error_in_ppm]
    """
    
    data = np.zeros(shape=(points.shape[0],4))
    lcs = np.zeros(shape=(points.shape[0],base.shape[0]))
    _star = star
    _planet = planet
    _system = ebm.System(_star,_planet)
    
    if (_planet.orbit.e != 0):
        phaseBaseline = _system.get_phase(basetimes).flatten()
        order = np.argsort(phaseBaseline)
        baselineLightcurve = base[order]
        phaseBaseline = phaseBaseline[order]
        
    for i in range(0, points.shape[0]):
                
        data[i,0] = points[i,0]
        data[i,1] = points[i,1]
        timeTotal = 0
        
        for j in range(0,loops):
            _star = star
            _planet = planet        
            _planet.map = ebm.Map.Map(nlat=points[i,1])
            _system = ebm.System(_star, _planet)


            tInt = time.time()

            Teq = _system.get_teq()
            T0 = np.ones_like(_system.planet.map.values)*Teq
            t0 = 0.
            t1 = t0+_system.planet.Porb
            dt = _system.planet.Porb/points[i,0]
            testTimes, testMaps = _system.run_model(T0, t0, t1, dt, verbose=False)
            if (_planet.orbit.e != 0):
                T0 = testMaps[-1]
                t0 = testTimes[-1]
                t1 = t0+_system.planet.Porb
                dt = _system.planet.Porb/points[i,0]
                testTimes, testMaps = _system.run_model(T0, t0, t1, dt, verbose=False, intermediates=True)
                testLightcurve = _system.lightcurve(testTimes, testMaps, bolo=False, wav=4.5e-6)

                phaseTest = _system.get_phase(testTimes).flatten()
                order = np.argsort(phaseTest)
                testLightcurve = testLightcurve[order]
                phaseTest = phaseTest[order]
                testLightcurve = np.interp(phaseBaseline, phaseTest, testLightcurve)
            else:
                testLightcurve = _system.lightcurve(bolo=False, wav=4.5e-6)

            tFin = time.time()
            timeTotal += (tFin - tInt)
        
        lcs[i] = testLightcurve
        data[i,3] = (1e6)*(np.amax(np.absolute(base - testLightcurve)))
        data[i,2] = (timeTotal/loops)*(1e3)

    return lcs, data


def Optimize(star, planet, error, verbose=False):
    """
    Optimizes spacial and temporal resolution for a given system
    
    Args:
        star (ebm.Star): The star to runs the tests on
        planet (ebm.Planet): The planet to run the tests on
        error (float): Amount of allowed error in ppm
        verbose (bool): If true will output progress as it computes
        
    Return:
        int: Optimized temporal resolution
        int: Optimized spacial resolution
    """
    _planet = planet
    _star = star
    aError = error

    #==========High Res Baseline Creation==========    
    if (verbose == True): 
        print("Starting baseline generation...")
    
    tInt = time.time()
    blt, blm, blc = CreateBaseline(_star, _planet)
    tFin = time.time()
    
    if (verbose == True): 
        print("Baseline generation complete; Time to Compute: " + str(round(tFin-tInt,2)) + "s")

    #===========Initial data creationg================
    space_points = 5
    temp_points = 5
    data = np.zeros(shape=((space_points*temp_points),4))
    for i in range (0, temp_points):
        for j in range (0, space_points):
            data[(i*space_points)+j,0]= ((i+1)*250)+0
            data[(i*space_points)+j,1] = ((j+1)*4)+0
    if (verbose == True): 
        print("First pass data points assigned")

    #==================First pass testing Area======================
    if (verbose == True): 
        print("Starting first pass...")
        
    tInt = time.time()
    lc, data = RunTests(_star, _planet, data, blc, blt)
    tFin = time.time()
    
    if (verbose == True): 
        print("First pass finished : Time to compute: " + str(round(tFin-tInt,2)) + "s")

    #=================First pass best point===================
    #print(data) #For debugging purposes 
    if (verbose == True):
        print("Processing first pass data...")
    iBest = None
    for i in range(0,space_points*temp_points):
        if (data[i,3]<=(aError*1.05)):
            if (iBest == None):
                iBest = i
            if(data[i,2] < data[iBest,2]):
                iBest = i
                
    #===========Second pass data creation================
    space_points = 5
    temp_points = 5
    dataDouble = np.zeros(shape=((space_points*temp_points),2))
    for i in range (0, temp_points):
        for j in range (0, space_points):
            dataDouble[(i*space_points)+j,0] = ((i)*50)+(data[iBest,0]-100)
            if (dataDouble[(i*space_points)+j,0]<100):
                dataDouble[(i*space_points)+j,0] = 100
            dataDouble[(i*space_points)+j,1] = ((j)*2)+(data[iBest,1]-4)
            if (dataDouble[(i*space_points)+j,1]<2):
                dataDouble[(i*space_points)+j,1] = 2
    if (verbose == True): 
        print("Second pass data points assigned")
    
    #==================Second pass testing Area======================
    if (verbose == True): 
        print("Starting second pass...")
        
    tInt = time.time()
    lc, dataDouble = RunTests(_star, _planet, dataDouble, blc, blt)
    tFin = time.time()
    if (verbose == True): 
        print("Second pass finished : Time to compute: " + str(round(tFin-tInt,2)) + "s")
    
    #=================Finding best second pass point===================
    #print(data) #For debugging purposes 
    if (verbose == True):
        print("Processing second pass data...")
    iBest = None
    for i in range(0,space_points*temp_points):
        if (dataDouble[i,3]<=aError):
            if (iBest == None):
                iBest = i
            if(dataDouble[i,2] < dataDouble[iBest,2]):
                iBest = i

    if (iBest == None):
        print("No points match requested error")
    else:
        print("Temporal: " + str(dataDouble[iBest,0]) + " Spacial: " + str(dataDouble[iBest,1]))
        print("Time for compute: " + str(round(dataDouble[iBest, 2],2)) +"ms : Error: " + str(round(dataDouble[iBest, 3],2)) + "ppm")
        print("Expected compute time @ 1,000,000 cycles: " + str((round((dataDouble[iBest, 2]*1e3/60)/60,2))) + " Hrs")
    
    temp = float(dataDouble[iBest,0])
    space = float(dataDouble[iBest,1])
    
    return temp, space