rem Prints file name without the path.
echo Running %~n0%~x0

set NEW_FILE=%1
set TARGET_FILE=%2

if not exist %NEW_FILE% (
	echo Can't find the source file: %NEW_FILE%
	exit /B 2
)

if not exist %TARGET_FILE% goto update

fc %TARGET_FILE% %NEW_FILE% > nul
if %errorlevel% == 1 goto update

echo %~n2%~x2 already up to date.
exit /B 0

:update
echo Updating %TARGET_FILE%.
copy /y %NEW_FILE% %TARGET_FILE%
