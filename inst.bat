@echo off
cls


msbuild DirectMLLib.sln /clp:ErrorsOnly /p:Configuration="Release" /p:Platform=x64 /t:restore /p:RestorePackagesConfig=true
msbuild DirectMLLib.sln /clp:ErrorsOnly /p:Configuration="Release" /p:Platform=x64 
call clbcall

call clbcall
del "Generated Files"\* /s /q 
del packages\* /s /q

git add *
git commit -m "API"
git push
