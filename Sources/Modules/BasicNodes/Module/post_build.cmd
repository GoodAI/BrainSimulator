rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3, $(Configuration) = %4

mkdir %1\Platform\BrainSimulator\bin\%4\modules\GoodAI.BasicNodes
xcopy /y /s %2%3*.* %1\Platform\BrainSimulator\bin\%4\modules\GoodAI.BasicNodes
xcopy /y %1..\BinaryLibs\GoodAiPlatformLibs\Nupic\nupic_core_solo.dll %1\Platform\BrainSimulator\bin\%4\modules\GoodAI.BasicNodes