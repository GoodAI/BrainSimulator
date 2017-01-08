rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3, $(Configuration) = %4, $(ProjectName) = %5

echo Running %5 post_build.cmd, config: %4

rem Only use when debugging.
rem echo Param 1 (SolutionDir) : %1
rem echo Param 2 (ProjectDir)  : %2
rem echo Param 3 (OutDir)      : %3
rem echo Param 4 (ConfigName)  : %4

if %4 == "Fast" (
	echo "Note: Fast build enabled, skipping module(s) copying."
	exit
)

set MODULE_NAME=BasicNodes
set TARGET_DIR=%2bin\%4\modules\GoodAI.%MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%

echo Copying module %MODULE_NAME%
xcopy /y /s /q %2..\..\Modules\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%
