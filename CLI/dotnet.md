# publish
`dotnet publish -c Release -r linux-x64 -f net7.0 --sc true`  
* `-c`  choose build configuration profile  
* `-r`  specify which os to publish  
> OS: linux, osx, win  
> Arch: x64, arm  
* `-f`  target framework  
* `--sc`  self-contained true | false  
> **self-contained** will compile all the .net to .dll  
> **framework dependent** will use .net runtime

