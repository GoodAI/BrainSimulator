rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3, $(Configuration) = %4

mkdir %2bin\%4\modules\GoodAI.BasicNodes
xcopy /y /s %2..\..\BasicNodes\Module\bin\%4\*.* %2bin\%4\modules\GoodAI.BasicNodes

mkdir %2bin\%4\modules\GoodAI.MNIST
xcopy /y /s %2..\..\MNIST\Module\bin\%4\*.* %2bin\%4\modules\GoodAI.MNIST
