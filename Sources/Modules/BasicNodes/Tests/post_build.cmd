rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3, $(Configuration) = %4, $(ProjectName) = %5

echo Running %5 post_build.cmd, config: %4

if %4 == "Fast" (
	echo "Note: Fast build enabled, skipping module(s) copying."
	exit
)

set MODULE_NAME=BasicNodes
set TARGET_DIR=%2bin\%4\modules\GoodAI.%MODULE_NAME%

echo Copying module %MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2..\..\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%


set MODULE_NAME=MNIST
set TARGET_DIR=%2bin\%4\modules\GoodAI.%MODULE_NAME%

echo Copying module %MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2..\..\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%
