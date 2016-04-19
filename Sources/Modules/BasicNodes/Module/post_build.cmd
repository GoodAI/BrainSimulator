set MODULE_NAME=GoodAI.BasicNodes

echo Running %MODULE_NAME% post_build.cmd

rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4


set BRAIN_SIM_DIR=%2..\..\..\Platform\BrainSimulator

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\%MODULE_NAME%


echo Creating directory: %TARGET_DIR%

mkdir %TARGET_DIR%


echo Copying from: & echo  %2%3 & echo to: & echo  %TARGET_DIR%.

xcopy /y /s %2%3*.* %TARGET_DIR%
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.

rem The files that are copied above are removed in case of clean (see BasicNodes.csproj, target "AfterClean").
