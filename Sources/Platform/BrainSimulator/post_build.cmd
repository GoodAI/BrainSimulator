rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3
mkdir %2%3ptx
echo xcopy /s /y %1CUDAKernels\%3ptx %2%3ptx
xcopy /s /y %1CUDAKernels\%3ptx %2%3ptx
xcopy /s /y %1..\Libraries\Caffe %2%3
mkdir %2%3modules\MNIST
xcopy /y %2%3MNIST.* %2%3modules\MNIST 
mkdir %2%3modules\XmlFeedForwardNet
xcopy /y %2%3XmlFeedForwardNet.* %2%3modules\XmlFeedForwardNet
rem xcopy /s /y %1CustomModels\bin\doc %2%3doc
rem xcopy /s /y %1BrainSimulator\bin\doc %2%3doc
xcopy /y %1..\Libraries\nupic.core\NupicCoreSoloWrapper\x64\Release\NupicCoreSoloWrapper\nupic_core_solo.dll %2%3