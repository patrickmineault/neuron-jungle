import vtk
import time

def run(neuron_num):
    file = "meshes-with-axons-filled-assembled/n%02d.vtp" % neuron_num

    readerVolume = vtk.vtkXMLPolyDataReader()
    readerVolume.SetFileName(file)
    readerVolume.Update()

    decimate = readerVolume
    nlod = 5

    for i in range(nlod):
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(decimate.GetOutputPort())
        smoother.SetNumberOfIterations(20)  # This has little effect on the error!
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetPassBand(.1)        # This increases the error a lot! .001
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.GenerateErrorScalarsOn()
        smoother.Update()

        decimate = vtk.vtkQuadricDecimation ()
        decimate.SetInputData(smoother.GetOutput())
        decimate.SetTargetReduction(.7)
        decimate.Update()


        full_out_name = 'meshes-with-axons-filled-assembled-lod/n%02d_LOD%d.obj' % (neuron_num, i - 1)
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(full_out_name)
        writer.SetInputData(decimate.GetOutput())
        writer.Write()

if __name__ == "__main__":
    for i in range(1, 97):
        print("Generating LOD for neuron %d" % i)
        run(i)
