cd build

del CMakeCache.txt

cmake -DMAKE_ONLY=BUILD_ALL  -G "Visual Studio 12 Win64"  ..

pause
