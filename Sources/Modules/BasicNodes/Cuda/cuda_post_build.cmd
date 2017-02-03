set MODULE_NAME=GoodAI.BasicNodes

echo Running %~n0%~x0 for module %MODULE_NAME% (CUDA), config: %4

rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4

set PLATFORM_DIR=%2..\..\..\Platform
set BRAIN_SIM_DIR=%PLATFORM_DIR%\BrainSimulator

if not exist %BRAIN_SIM_DIR% (
	echo Error: Target directory '%BRAIN_SIM_DIR%' not found. Please check paths.
	exit /B 2
)

set OUTPUT_DIR=%3

if not exist %OUTPUT_DIR% (
	echo Error: Output directory '%OUTPUT_DIR%' not found. Please check paths.
	exit /B 2
)

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\%MODULE_NAME%\ptx

if not exist %TARGET_DIR% mkdir %TARGET_DIR%

echo Copying module %MODULE_NAME% PTX files to: & echo  %TARGET_DIR%.

xcopy /y /s /q %OUTPUT_DIR%*.* %TARGET_DIR%
rem if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME% (CUDA)': copying successful.