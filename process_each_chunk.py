import dask
from dask.distributed import LocalCluster, Client
import h5py
import numpy as np
import os
import scipy
import scipy.ndimage
import shutil
import tempfile
import time
import vtk

def copy_and_replace(in_filename, out_filename, replacement=None):
    with open(in_filename, 'r') as f:
        data = f.read()

    if replacement is not None:
        data = data.format(replacement)

    with open(out_filename, 'w') as f:
        f.write(data)

@dask.delayed
def process_one_chunk(filename, out_filename):
    print(f"Processing file {filename}")
    xdmf_template = "chunk_template.xdmf"
    _, basename = os.path.split(filename)
    xdmf_out = f"xdmf/{basename}.xdmf"

    copy_and_replace(xdmf_template, xdmf_out, replacement='../' + filename)

    cube = h5py.File(filename, 'r')
    neuron_ids = np.array(cube['neuron_ids'])
    cube.close()

    # Prepare to read the file.
    readerVolume = vtk.vtkXdmfReader()
    readerVolume.SetFileName(xdmf_out)
    readerVolume.Update()

    # Extract the region of interest.
    # voi = vtk.vtkExtractVOI()
    # voi.SetInputConnection(readerVolume.GetOutputPort())
    # voi.SetVOI(0, 1023, 0, 1023, 0, 1023)
    # voi.SetSampleRate(1, 1, 1)
    # voi.Update()  # Necessary for GetScalarRange().

    for index in neuron_ids:
        full_out_name = out_filename % index
        if os.path.exists(full_out_name):
            print("Skipping neuron %d" % index)
            continue
        print("Processing neuron %d" % index)

        # Prepare surface generation.
        contour = vtk.vtkDiscreteMarchingCubes()  # For label images.
        contour.SetInputConnection(readerVolume.GetOutputPort())
        contour.SetValue(0, index)
        contour.Update()  # Needed for GetNumberOfPolys()!!!

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(15)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(.01)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.GenerateErrorScalarsOn()
        smoother.Update()

        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(smoother.GetOutput())
        decimate.SetTargetReduction(.8)
        decimate.PreserveTopologyOn()
        decimate.BoundaryVertexDeletionOff()
        decimate.Update()

        smoothed_polys = decimate.GetOutput()

        writer = vtk.vtkXMLDataSetWriter()
        writer.SetFileName(full_out_name)
        writer.SetInputData(smoothed_polys)
        writer.Write()

    return 1

@dask.delayed
def infill_one_chunk(filename_in, filename_out):
    f = h5py.File(filename_in, 'r')
    A = np.array(f['data'])
    neuron_ids = np.array(f['neuron_ids'])
    A = scipy.ndimage.morphology.grey_erosion(scipy.ndimage.morphology.grey_dilation(A, size=5), size=5)

    cube = h5py.File(filename_out, 'w')
    cube.create_dataset('data', A.shape, compression="gzip", data=A)
    cube.create_dataset('neuron_ids', neuron_ids.shape, data=neuron_ids)
    cube.close()
    return 1

if __name__ == '__main__':
    # cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    # client = Client(cluster, heartbeat_interval=30000)
    # r = 0
    # for i in range(6):
    #     for j in range(9):
    #         for k in range(4):
    #             print(f'Processing chunk {i}, {j}, {k}')
    #             r += infill_one_chunk(f'neuron-volume-with-axons-unfilled/x{i}y{j}z{k}.hdf5',
    #                                   f'neuron-volume-with-axons-filled/x{i}y{j}z{k}.hdf5')
    # r.compute()
    # cluster.close()
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster, heartbeat_interval=30000)

    r = 0
    for i in range(6):
        for j in range(9):
            for k in range(4):
                print(f'Processing chunk {i}, {j}, {k}')
                r += process_one_chunk(f'neuron-volume-with-axons-filled/x{i}y{j}z{k}.hdf5',
                                       f'meshes-with-axons-filled/x{i}y{j}z{k}_n%03d.vtp')
    r.compute()
#time.sleep(10)
