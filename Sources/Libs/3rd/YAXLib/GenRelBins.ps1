# You do NOT need to run this script. 
# This is for creating the binaries signed 
# with another key, which is only available
# in the main developer's machine. 
# Although it's harmless, running this by 
# others would fail this very custom build.

param (
    [string] $key = "../../_SinaKey/Sina.Libs.pfx",
    [string] $src = "YAXLib/YAXLib.csproj",
    [string] $dst = "YAXLib/~deploy~.msbuild")

$msb = "$env:windir\Microsoft.NET\Framework\v4.0.30319\msbuild.exe"
if(!(Test-Path $msb))
{
    $msb = "$env:windir\Microsoft.NET\Framework\v3.5\msbuild.exe"
    if(!(Test-Path $msb))
    {
        Write-Error "MSBuild not found!"
        exit 1
    }
}

$scrPath = $MyInvocation.MyCommand.Definition
$scrDir = [System.IO.Path]::GetDirectoryName($scrPath)

$src = [System.IO.Path]::Combine($scrDir, $src)
if(!(Test-Path $src))
{
    Write-Error "The specified source file not found!"
    exit 1
}

$dst = [System.IO.Path]::Combine($scrDir, $dst)
$dstDir = [System.IO.Path]::GetDirectoryName($dst)

$fullkey = [System.IO.Path]::Combine($dstDir, $key)
if(!(Test-Path $fullkey))
{
    Write-Error "The specified key file not found!"
    exit 1
}

Copy-Item $src $dst

# now read and replace patterns in dst

$patSign = "\<\s*SignAssembly\s*\>(?<value>[^\<]+)\<\/\s*SignAssembly\s*\>"
$patKey = "\<\s*AssemblyOriginatorKeyFile\s*\>(?<value>[^\<]+)\<\/\s*AssemblyOriginatorKeyFile\s*\>"
$patOut = "\<\s*OutputPath\s*\>(?<value>[^\<]+)\<\/\s*OutputPath\s*\>"
$patDoc = "\<\s*DocumentationFile\s*\>(?<value>[^\<]+)\<\/\s*DocumentationFile\s*\>"

$regexSign = [regex]$patSign
$regexKey = [regex]$patKey
$regexOut = [regex]$patOut
$regexDoc = [regex]$patDoc

$text = [System.IO.File]::ReadAllText($dst)
$text = $regexSign.Replace($text, "<SignAssembly>true</SignAssembly>")
$text = $regexKey.Replace($text, "<AssemblyOriginatorKeyFile>$key</AssemblyOriginatorKeyFile>")
$text = $regexOut.Replace($text, 
    {
        param($m)
        $newPath = [System.IO.Path]::Combine($m.Groups["value"].Value, "Signed")
        return "<OutputPath>$newPath</OutputPath>"
    })
$text = $regexDoc.Replace($text, 
    {
        param($m)
        $docOld = $m.Groups["value"].Value
        $docDir = [System.IO.Path]::GetDirectoryName($docOld)
        $docFile = [System.IO.Path]::GetFileName($docOld)
        $newPath = [System.IO.Path]::Combine([System.IO.Path]::Combine($docDir, "Signed"), $docFile)
        return "<DocumentationFile>$newPath</DocumentationFile>"
    })


[System.IO.File]::WriteAllText($dst, $text)

# changes directory to the specified location, while allowing
# to return back to the original location with Pop-Location
Push-Location $dstDir

$dstFileName = [System.IO.Path]::GetFileName($dst)
& "$msb" "$dstFileName"

Pop-Location

Remove-Item $dst
