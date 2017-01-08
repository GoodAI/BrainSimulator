rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(Configuration) = %4, $(ProjectName) = %5

set MODULE_NAME=%5

echo Running %MODULE_NAME% post_build.cmd

set TARGET_DIR=%2%3modules
if not exist %TARGET_DIR% mkdir %TARGET_DIR%

set TARGET_DIR=%2%3Licenses
if not exist %TARGET_DIR% mkdir %TARGET_DIR%

echo Copying licences to: & echo  %2%3\Licenses

xcopy /y /s /q %2..\..\..\Licenses\*.* %2%3\Licenses
if %ERRORLEVEL% GEQ 1 exit /B 3

echo Module '%MODULE_NAME%' post_build: copying successful.