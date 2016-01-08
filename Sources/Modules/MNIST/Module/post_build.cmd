set MODULE_NAME=GoodAI.MNIST

set BRAIN_SIM_DIR=%2..\..\..\Platform\BrainSimulator

echo Running %MODULE_NAME% post_build.cmd
rem echo Param 1 (SolutionDir): %1 (not used)
rem echo Param 2 (ProjectDir) : %2
rem echo Param 3 (OutDir)     : %3
rem echo Param 4 (ConfigName) : %4

if not exist %BRAIN_SIM_DIR% (
	echo Error: Target directory %BRAIN_SIM_DIR% not found. Please check paths.
	exit /B 2
)

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\%MODULE_NAME%

echo Copying from: & echo  %2%3 & echo to: & echo  %TARGET_DIR%.

if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2%3*.* %TARGET_DIR%
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.
