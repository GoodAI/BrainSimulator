@echo off

if "%~1"=="" goto ARGUMENT_ERROR
if "%~2"=="" goto ARGUMENT_ERROR

set INSTALLER_NAME_BASE=..\bin\Release\BrainSimulatorInstaller-
set INSTALLER_FILE=%INSTALLER_NAME_BASE%vX.Y.Z-00-unsigned.msi
set CERT_FILE=%2
set DESCRIPTION="GoodAI Brain Simulator Installer"

echo Params
echo   Version and build number: %1
echo   Installer file          : %INSTALLER_FILE%
echo   Certificat file         : %CERT_FILE%
echo   Description             : %DESCRIPTION%

set /p PASSWORD=Enter the certificate password (WARNING: will be echoed):

set INSTALLER_FOR_SIGNING="%INSTALLER_NAME_BASE%%~1-to-be-signed.msi"
copy %INSTALLER_FILE% %INSTALLER_FOR_SIGNING%

"C:\Program Files (x86)\Windows Kits\8.1\bin\x64\signtool.exe" sign /fd SHA256 /a /f %CERT_FILE% /p %PASSWORD% /d %DESCRIPTION% %INSTALLER_FOR_SIGNING%

move %INSTALLER_FOR_SIGNING% "%INSTALLER_NAME_BASE%%~1.msi"

rem Long live spaghetti code !-)
goto DONE

:ARGUMENT_ERROR

echo.
echo Usage: %0 "<version>" "<certificate-file>"
echo.
echo The first argument must be version and build number (preferably in format "vX.Y.Z-WW")
echo The second argument must be full name of a code signing certificate (*.pfx)
echo.

:DONE