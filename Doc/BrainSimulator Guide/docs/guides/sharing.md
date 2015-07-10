# How to share a module

* The easiest way to share a module is to share its source code using [GitHub](https://github.com/). You just create a repository, put all your code in there, write some notes for the users and post a link to your repository on the [Brain Simulator forum](http://forum.goodai.com/index.php?forums/brain-simulator-shared-projects.16/).
* If for some reason you do not want to share the source code you may use the [Brain Simulator forum](http://forum.goodai.com/index.php?forums/brain-simulator-shared-projects.16/) to upload zipped binaries.

## Sharing the source code

Sharing a source code is a very easy and straightforward. You just need a place to upload the contents of you Project folder (the folder with the .sln file ;-)). 
We suggest GitHub as it makes it really easy to update the project as it evolves and makes all the coding very comfy... 

Then you just let others know that there is a new module available on the [Brain Simulator forum](http://forum.goodai.com/index.php?forums/brain-simulator-shared-projects.16/) by sharing a link to your repository. 

## Building and installing from source

* Clone the repository or unpack the source files to any folder.
* Build the solution using Visual Studio (You will need [CUDA 7](https://developer.nvidia.com/cuda-downloads) installed. In case you installed the Brain Simulator to a custom path you need to correct it in the project properties and references.).
* Find the finished binaries (`<ModuleProjectFolder>\Module\bin\<Debug|Release>`) and copy everything (including ptx and res folders if present) into `<YourBrainSimulatorInstallation>\modules\ModuleName\`. For example installing the NewModule example into the default Brain Simulator install location  would mean copying the files to `c:\Program Files\GoodAI\Brain Simulator\modules\NewModule\`.
* Note that the module's .dll file must have the same name as the folder it's in.
* Restart Brain Simulator and check the Console. You should get an information that your module has been successfully loaded. (e.g. `Module loaded: NewModule.dll (version=1)`) 

## Sharing the binaries only

* Build the binaries
* Create an archive with a folder named after the module's main .dll and put all the files from your bin/debug|release folder into it. For example the NewModule.dll will be packed as NewModule\NewModule.dll.  
* Upload the file to any service you fancy and put a link to the [Brain Simulator forum](http://forum.goodai.com/index.php?forums/brain-simulator-shared-projects.16/) or upload the file to the forum directly.
* Please note that downloading .dll files is considered _highly_ unsafe so you may encounter some hesitation about downloading your module.

## Installing from binaries

* Please note that downloading and using .dll files from unknown authors is _risky_ so make sure you are getting the module from a trustworthy author.
* Unpack all the files from the archive into `<YourBrainSimulatorInstallation>\modules\` and make sure that the main module .dll is at its correct location (`<YourBrainSimulatorInstallation>\modules\<ModuleName>\<ModuleName.dll>`). If not, move all the files to the proper location.
* Restart Brain Simulator and check the Console. You should get an information that your module has been successfully loaded. (e.g. `Module loaded: NewModule.dll (version=1)`)
