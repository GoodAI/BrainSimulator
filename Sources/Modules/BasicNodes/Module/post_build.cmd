set MODULE_NAME=GoodAI.BasicNodes

echo Running %MODULE_NAME% post_build.cmd, config: %4

rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4

set BRAIN_SIM_DIR=%2..\..\..\Platform\BrainSimulator

if not exist %BRAIN_SIM_DIR% (
	echo Error: Target directory %BRAIN_SIM_DIR% not found. Please check paths.
	exit /B 2
)

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\%MODULE_NAME%

if not exist %TARGET_DIR% mkdir %TARGET_DIR%

rem Debug version: echo Copying from: & echo  %2%3 & echo to: & echo  %TARGET_DIR%.
echo Copying module %MODULE_NAME% files to: & echo  %TARGET_DIR%.

xcopy /y /s /q %2%3*.* %TARGET_DIR%
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.

rem The files that are copied above are removed in case of clean (see BasicNodes.csproj, target "AfterClean").
