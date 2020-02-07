
import glob
import os
import vtk

def assemble_neuron(neuron_num):
    print(f"Processing neuron {neuron_num}")
    files = glob.glob("meshes-with-axons-filled/*_n%03d.vtp" % neuron_num)
    out_filename = "meshes-with-axons-filled-assembled/n%02d.vtp" % neuron_num

    appendFilter = vtk.vtkAppendPolyData()
    for file in files:
        _, basename = os.path.split(file)

        x, y, z = int(basename[1]), int(basename[3]), int(basename[5])

        print(x, y, z)

        # Prepare to read the file.
        readerVolume = vtk.vtkXMLPolyDataReader()
        readerVolume.SetFileName(file)
        readerVolume.Update()

        # Set up the transform filter
        translation = vtk.vtkTransform()
        translation.Translate(x * 1023.0, y * 1023.0, z * 1023.0 * 2.49)

        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetInputConnection(readerVolume.GetOutputPort())
        transformFilter.SetTransform(translation)
        transformFilter.Update()

        appendFilter.AddInputData(transformFilter.GetOutput())

    appendFilter.Update()

    writer =  vtk.vtkXMLDataSetWriter()
    writer.SetFileName(out_filename)
    writer.SetInputData(appendFilter.GetOutput())
    writer.Write()

if __name__ == '__main__':
    for i in range(1, 97):
        assemble_neuron(i)
