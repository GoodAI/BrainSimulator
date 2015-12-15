rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3, $(Configuration) = %4

mkdir %2..\..\..\Platform\BrainSimulator\bin\%4\modules\GoodAI.BasicNodes
xcopy /y /s %2%3*.* %2..\..\..\Platform\BrainSimulator\bin\%4\modules\GoodAI.BasicNodes

mkdir %2..\..\..\Platform\CoreTests\bin\%4\modules\GoodAI.BasicNodes
xcopy /y /s %2%3*.* %2..\..\..\Platform\CoreTests\bin\%4\modules\GoodAI.BasicNodes
