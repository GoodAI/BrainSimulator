rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4

echo Running ToyWorldConversation post_build.cmd
echo Param 1 (SolutionDir): %1
echo Param 2 (ProjectDir) : %2
echo Param 3 (OutDir)     : %3
echo Param 4 (ConfigName) : %4

set BRAIN_SIM_DIR=%2..\..\..\Platform\BrainSimulator

if not exist %BRAIN_SIM_DIR% (
	echo "Error: Target directory not found. Please check paths."
	exit /B 2
)

set TARGET_DIR=%BRAIN_SIM_DIR%\bin\%4\modules\ToyWorldConversation
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s %2%3*.* %TARGET_DIR%