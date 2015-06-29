rem $(ProjectDir) = %1, $(ConfigurationName) = %2

rem NOTES:
rem This script assumes that CUDAKernels have been build with the same configuration (Debug/Release) as this project
rem And that CUDAKernels have been built before this project.
rem 
rem If you don't have a clean checkout, it is recommended to purge CUDAKernels/bin directory before rebuilding 
rem CUDAKernels (for the purpose of creating fresh installation package). This bin directory tends to accumulate
rem garbage over time and everything present in the source 'ptx' subdirectory will be packed to ptx.zip

rem Change current dir to the parent of ptx dir to get the right directory structure inside the zip file

echo %1
echo %2

cd %1..\..\Modules\BasicNodes\Module\bin\%2

set ZIPFILE=%1bin\ptx.zip
if exist %ZIPFILE% del %ZIPFILE%
call %1Tools\zip.exe -r %ZIPFILE% ptx