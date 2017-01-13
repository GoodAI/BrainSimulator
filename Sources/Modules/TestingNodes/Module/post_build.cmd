set MODULE_NAME=GoodAI.TestingNodes

rem If you modify this file, please rebuild the project or delete obj/*.pdb to cause rebuild.
rem Becase this file is considered an input file and might cause eternal unnecessary rebuilds.

echo Running %MODULE_NAME% post_build.cmd, config: %4

rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4

set PLATFORM_DIR=%2..\..\..\Platform
set BRAIN_SIM_DIR=%PLATFORM_DIR%\BrainSimulator

if not exist %BRAIN_SIM_DIR% (
	echo Error: Target directory %BRAIN_SIM_DIR% not found. Please check paths.
	exit /B 2
)

call %PLATFORM_DIR%\Core\update_file.cmd "%~2\bin\doc_new.xml" "%~2\conf\doc.xml"
if %ERRORLEVEL% GEQ 1 exit /B 4

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\%MODULE_NAME%

if not exist %TARGET_DIR% mkdir %TARGET_DIR%

rem Debug version: echo Copying from: & echo  %2%3 & echo to: & echo  %TARGET_DIR%.
echo Copying module %MODULE_NAME% files to: & echo  %TARGET_DIR%.

xcopy /y /s /q %2%3*.* %TARGET_DIR%
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.
