# -*- coding: utf-8 -*-
"""
Connect to the different quantum computers
"""
#%% Import libraries
import qiskit as qis
import numpy as np

import qiskit_braket_provider as qisbra # AWS
import braket.aws as aws # To retrieve AWS jobs with aws.AwsQuantumTask(arn)
import qiskit_ibm_runtime as qisibm # IBM
from qiskit_ibm_runtime import SamplerV2 as IBM_Sampler
import matplotlib.pyplot as plt


# For Aquila
from braket.ahs.atom_arrangement import AtomArrangement
from braket.timings.time_series import TimeSeries
from braket.ahs.driving_field import DrivingField
from braket.timings.time_series import TimeSeries
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.devices import LocalSimulator # Debugging
from collections import Counter # Visualize results

# To remove transpiler warning for IonQ (we already know Qiskit uses level 2 instead of 1)
import warnings
from qiskit_ionq.exceptions import IonQTranspileLevelWarning
warnings.filterwarnings("ignore", category=IonQTranspileLevelWarning)

"""
Selected computers

IBM: ibm-kingston (Uses Heron R2)
AWS:
    QuEra: Aquila (based on laser-trapped neutral atoms, 256 qubits). us-east-1. $0.30/task + $0.01/shot (on demand)
        https://us-west-2.console.aws.amazon.com/braket/home?region=us-west-2#/devices/arn:aws:braket:us-east-1::device/qpu/quera/Aquila
    IonQ: Forte Enterprise 1 (trapped ions)
        https://us-west-2.console.aws.amazon.com/braket/home?region=us-west-2#/devices/arn:aws:braket:us-east-1::device/qpu/ionq/Forte-Enterprise-1
    Rigetti: Cepheus™-1-108Q $0.30/task + $0.000425/shot (on demand).
     <<CZ gates are more robust against phase errors, which are common in superconducting systems>>
        https://us-west-2.console.aws.amazon.com/braket/home?region=us-west-2#/devices/arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q
    AWS itself:

Setup:
https://aws.amazon.com/es/blogs/quantum-computing/setting-up-your-local-development-environment-in-amazon-braket/
Step 2:
https://aws.amazon.com/es/blogs/quantum-computing/introducing-the-qiskit-provider-for-amazon-braket/
Braket:
    https://us-west-2.console.aws.amazon.com/braket/home?region=us-west-2#/dashboard
"""
#%% Load Providers
print("Loading and Connecting to dependencies of Computers...")
#%%% This section is needed when this code is executed for the first time on your PC.
# Store your credentials in an .env file with these keywords in the root directory of the code
"""
qisibm.QiskitRuntimeService.save_account(
    token= PUT YOUR TOKEN HERE,
    instance= PUT YOUR INSTANCE HERE,
    set_as_default = True,
    overwrite = True
    )
"""
#%%%
IBMProvider = qisibm.QiskitRuntimeService()
AWSProvider = qisbra.BraketProvider()

print("Dependencies Loaded!")
print("-"*40)
#%% Functions
def Status():
    """
    Check the availability of all computers and return their backends
    """
    print("IBM COMPUTERS:\n")
    IBMBackends = IBMProvider.backends(simulator = False)
    for i in range(len(IBMBackends)):
        print("Name:", IBMBackends[i].name)
        if IBMBackends[i] == None:
            print("Cant access status")
        else:
            print('Status:')
            print('  Operational: ', IBMBackends[i].status().operational)
            print('  Pending jobs:', IBMBackends[i].status().pending_jobs)
            print('  Status message:',IBMBackends[i].status().status_msg)
        print("-" * 40)
    print("For a full description of each QC, check: \n https://quantum.cloud.ibm.com/computers \n")
    print("*"*40)

    print("AWS COMPUTERS:\n")
    AWSBackends = AWSProvider.backends()

    for i in range(len(AWSBackends)):
        print("Name:", AWSBackends[i].name)
        print("-" * 40)
    print("For a full description of each QC, check: \n https://quantum.cloud.ibm.com/computers \n")
    print("*"*40)

    print("Returned tuple containing the lists of available QCs in the printed order.")
    return IBMBackends,AWSBackends

def CallIBM(qc,backend,shots, lvl = 1,drawTranspile = False):
    """
    Call an IBM QC for a job.
    https://quantum.cloud.ibm.com/docs/es/guides/hello-world
    """
    # Get backend
    if type(backend) == str:
        backend = IBMProvider.backend(backend)

    print(f"Preparing IBM job at backend {backend.name}")
    # First, optimize the circuit for the selected backend
    qcT = qis.compiler.transpile(qc,backend = backend, optimization_level= lvl) # It is exactly the same as generate_preset_pass_manager
    #https://quantum.cloud.ibm.com/docs/es/guides/visualize-circuits For saving circuits as PDF
    if drawTranspile:
        qcT.draw("mpl",idle_wires=False)

    # Then create and send the job
    # Retrieve job details
    sampler = IBM_Sampler(mode = backend)
    sampler.options.default_shots = shots
    job = sampler.run([qcT])

    print(f"Job created and returned as a variable. Job ID: {job.job_id()}")
    print("Jobs can be checked at \n https://eu-de.quantum.cloud.ibm.com/workloads")
    return job


#%% AWS
def CallAWS(qc,backend,shots, lvl = 1, drawTranspile = False):
    """
    Call an AWS QC for a job
    Region options:
       us-west-1 (Rigetti)
       us-east-1 (QuEra, IonQ)
    """
    # Get backend
    if type(backend) == str:
        backend = AWSProvider.get_backend(backend)
    print(f"Preparing AWS job at backend {backend.name}")
    # First, optimize the circuit for the selected backend
    qcT = qis.compiler.transpile(qc,backend = backend, optimization_level=lvl) # It is exactly the same as generate_preset_pass_manager
    qcT.data = [inst for inst in qcT.data
    if inst[0].name != "barrier"] # Remove barriers (not supported by IonQ or QuEra)
    #https://quantum.cloud.ibm.com/docs/es/guides/visualize-circuits For saving circuits as PDF
    if drawTranspile:
        qcT.draw("mpl",idle_wires=False)

    # Then create and send the task
    # Retrieve task details
    task = backend.run(qcT,shots = shots)#,memory = True)

    print(f"Job created and returned as a variable. Job ID: {task.job_id()}")
    print("Jobs can be checked at \n https://us-west-1.console.aws.amazon.com/braket/home?region=us-west-1#/tasks  \n ")
    return task

#%% Aquila
def Visualize_Driving_Field(drive):
    # https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    time_series = drive.amplitude.time_series
    ax.plot(time_series.times(), time_series.values(), '.-')
    ax.grid()
    ax.set_ylabel('Omega [rad/s]')

    ax = axes[1]
    time_series = drive.detuning.time_series
    ax.plot(time_series.times(), time_series.values(), '.-')
    ax.grid()
    ax.set_ylabel('Delta [rad/s]')

    ax = axes[2]
    time_series = drive.phase.time_series
    # Note: The phase time series is interpreted as a piecewise constant function
    ax.step(time_series.times(), time_series.values(), '.-', where='post')
    ax.set_ylabel('phi [rad]')
    ax.grid()
    ax.set_xlabel('time [s]')

    plt.show()
    return

def CallAquila(nQbits,shots, showDist = False, showField = False, simulate = False):
    # https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html
    # TODO: Adapt this because a large part should be moved to MyCircuits
    # Prepare the atom grid
    grid = AtomArrangement()
    """
    From the Bachelor's thesis calculations, we know that the distance between atoms can be computed as

    d = W/sqrt(nQbits) (µm, if W is in µm)

    Where W = 75 µm is the maximum grid width (height ≈ width)

    The number of qubits must be even
    """
    W = 75 * 10**(-6)  # Maximum grid width (m)     H = 75 µm # Maximum grid height (actually 76)
    d = W/np.sqrt(nQbits) # In meters
    nFilas = int(np.ceil(W/d)) # Number of qubits per row
    nCol = nFilas # Number of qubits per column
    # Place atoms one by one
    aux = 0 # Counter for the placed qubits
    for i in range(nFilas):
        for j in range(nCol): # Fill column by column
            if aux < nQbits:
                grid.add([d*i, d*j])
                aux += 1
    # Display the distribution if requested
    if showDist:
        fig,ax = plt.subplots()
        xs, ys = [grid.coordinate_list(dim) for dim in (0, 1)]
        # print(len(xs))
        ax.plot(xs, ys, 'r.', ms=15)
        for idx, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y, f" {idx}", fontsize=12)
        fig.show()

    # Grid created, initialize variables
    omega = 1.58 * 10**6 # Constant float
    #omega = omega*10  # Maximum frequency, if we want to use it

    Omega = TimeSeries() # Time series

    Delta = TimeSeries()
    Delta.put(0.,0.)

    Phi = TimeSeries()
    Phi.put(0.,0.)

    endTime = np.pi/(omega) # Seconds. Obtained by finding t such that Omega * t = pi/2
    """
    dt = 5.0 * 10**(-8) # Minimum value accepted by Aquila
    times = np.arange(0,endTime + dt,dt) # For some reason it reports that pulse durations are not equal
    """
    times = np.linspace(0,endTime,30) # It may seem like few points, but the interval is very small
    y = omega * np.sin(np.pi * times/endTime)**2
    y[-1] = 0.
    for i in range(len(times)):
        Omega.put(times[i],y[i])

    Delta.put(endTime,0.)
    Phi.put(endTime,0.)

    drive = DrivingField(amplitude = Omega, phase = Phi, detuning = Delta)
    if showField:
        Visualize_Driving_Field(drive)

    # Close the AHS program
    program = AnalogHamiltonianSimulation(register=grid, hamiltonian=drive)

    # Create and submit the task
    if simulate:
        aquila = LocalSimulator("braket_ahs") # Local
        task = aquila.run(program,shots = shots)
    else:
        aquila = aws.AwsDevice("arn:aws:braket:us-east-1::device/qpu/quera/Aquila")
        program_discretized = program.discretize(aquila) # Round values so Aquila can process them
        task = aquila.run(program_discretized,shots = shots)
        print(f"Job created and returned as a variable. Job ID: {task.id}")
        print("Jobs can be checked at \n https://us-west-1.console.aws.amazon.com/braket/home?region=us-west-1#/tasks  \n ")
    return task
