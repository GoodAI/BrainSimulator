rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3

rem Creating the modules directory...

set TARGET_DIR=%2%3\modules
if not exist %TARGET_DIR% mkdir %TARGET_DIR%

rem Copying modules...

set MODULE_NAME=BasicNodes
set TARGET_DIR=%2%3\modules\GoodAI.%MODULE_NAME%
echo Copying module %MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2..\..\Modules\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%

if not exist %2..\..\Modules\InternalNodes goto SKIPINTERNALNODES

set MODULE_NAME=InternalNodes
set TARGET_DIR=%2%3\modules\GoodAI.%MODULE_NAME%
echo Copying module %MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2..\..\Modules\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%

:SKIPINTERNALNODES

set MODULE_NAME=MNIST
set TARGET_DIR=%2%3\modules\GoodAI.%MODULE_NAME%
echo Copying module %MODULE_NAME%
if not exist %TARGET_DIR% mkdir %TARGET_DIR%
xcopy /y /s /q %2..\..\Modules\%MODULE_NAME%\Module\bin\%4\*.*  %TARGET_DIR%
