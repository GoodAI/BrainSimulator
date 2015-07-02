
rem $(ProjectDir) = %1, $(ConfigurationName) = %2

rem NOTES:
rem This script assumes that CUDA kernels have been build with the same configuration (Debug/Release) as this project
rem And that the CUDA project has been built before this project.
rem 
rem If you don't have a clean checkout, it is recommended to purge the source bin/ptx directory before rebuilding 
rem CUDA kernels (for the purpose of creating fresh installation package). This bin directory tends to accumulate
rem garbage over time and everything present in the source 'ptx' subdirectory will be packed to ptx.zip

rem Change current dir to the parent of ptx dir to get the right directory structure inside the zip file

echo Param 1 (ProjectDir): %1
echo Param 2 (ConfigName): %2

rem *** BasicNodes ***

cd %1..\..\Modules\BasicNodes\Module\bin\%2
echo %cd%
set TARGETDIR=..\..\..\..\..\Installer\InstallerAction\bin\modules\GoodAI.BasicNodes

if not exist %TARGETDIR% mkdir %TARGETDIR%
set ZIPFILE=%TARGETDIR%\ptx.zip
if exist %ZIPFILE% del %ZIPFILE%
call %1\Tools\zip.exe -r %ZIPFILE% ptx


rem *** XmlFeedForwardNet ***

cd %1..\..\Modules\XmlFeedForwardNet\Module\bin\%2
echo %cd%
set TARGETDIR=..\..\..\..\..\Installer\InstallerAction\bin\modules\GoodAI.XmlFeedForwardNet

if not exist %TARGETDIR% mkdir %TARGETDIR%
set ZIPFILE=%TARGETDIR%\ptx.zip
if exist %ZIPFILE% del %ZIPFILE%
call %1\Tools\zip.exe -r %ZIPFILE% ptx
