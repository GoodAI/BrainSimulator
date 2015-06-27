rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3
mkdir %2%3modules

rem mkdir %2%3modules\MNIST
rem xcopy /y %2%3MNIST.* %2%3modules\MNIST 
rem mkdir %2%3modules\XmlFeedForwardNet
rem xcopy /y %2%3XmlFeedForwardNet.* %2%3modules\XmlFeedForwardNet
rem xcopy /s /y %1CustomModels\bin\doc %2%3doc
rem xcopy /s /y %1BrainSimulator\bin\doc %2%3doc
rem xcopy /y %1..\Libraries\nupic.core\NupicCoreSoloWrapper\x64\Release\NupicCoreSoloWrapper\nupic_core_solo.dll %2%3