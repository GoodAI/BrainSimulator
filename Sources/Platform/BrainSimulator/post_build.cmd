rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3

rem Creating the modules directory...
mkdir %2%3modules

mkdir %2%3Licenses
xcopy /y /s %2..\..\..\Licenses\*.* %2%3\Licenses