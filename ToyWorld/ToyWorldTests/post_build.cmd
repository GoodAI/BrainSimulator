rem $(SolutionDir) = %1, $(ProjectDir) = %2, $(OutDir) = %3 $(TargetFileName) = %4

echo Running ToyWorld post_build.cmd
echo Param 1 (SolutionDir): %1
echo Param 2 (ProjectDir) : %2
echo Param 3 (OutDir)     : %3
echo Param 4 (TargetFile) : %4

%~2..\..\..\packages\OpenCover.4.6.519\tools\OpenCover.Console.exe -register:user -target:"%~2..\..\..\packages\xunit.runner.console.2.1.0\tools\xunit.console.exe" -targetargs:"%~2%~3%~4 -noshadow -notrait \"coverageSkip=true\"" -filter:"-[xunit*]* -[TypeMap]* -[*Tests]* +[*]*" -output:%~2coverage.xml

%~2..\..\..\packages\ReportGenerator.2.4.4.0\tools\ReportGenerator.exe "-reports:%~2coverage.xml" "-targetdir:%~2\coverage" "-historydir:%~2\coveragehistory"