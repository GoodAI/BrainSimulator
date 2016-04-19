set MODULE_NAME=GoodAI.BrainSimulator

echo Running %MODULE_NAME% post_build.cmd

rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4


echo Creating the directory: %2%3modules

mkdir %2%3modules


echo Creating the directory: %2%3Licenses

mkdir %2%3Licenses


echo Copying from: & echo  %2..\..\..\Licenses & echo to: & echo  %2%3\Licenses

xcopy /y /s %2..\..\..\Licenses\*.* %2%3\Licenses
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.