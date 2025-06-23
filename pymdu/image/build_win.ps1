# ⚙️ Configuration
$pythonPath = "C:\Users\bbrangeon\miniforge3\envs\pymdu\python.exe"  # ou le chemin complet vers python.exe
$vcvarsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
 
# Récupération des includes Pybind11
$pybindIncludes = & $pythonPath -m pybind11 --includes
$extSuffix = & $pythonPath -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
$pythonInclude = & $pythonPath -c "import sysconfig; print(sysconfig.get_paths()['include'])"
$pythonLib = & $pythonPath -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"

echo $pythonLib 
# Commande de compilation
$compileCommand = @"
cl /LD /O2 /std:c++17 lib.cpp -IC:\Users\bbrangeon\miniforge3\envs\pymdu\Include -IC:\Users\bbrangeon\miniforge3\envs\pymdu\Lib\site-packages\pybind11\include -IC:C:\Users\bbrangeon\miniforge3\envs\pymdu\include /link /DLL /LIBPATH:C:\Users\bbrangeon\miniforge3\envs\pymdu\libs  /Fe:rasterize$extSuffix `
"@
 
 # 🛠️ Construction de la commande de compilation
$compileCommand2 = @"
cl /LD /O2 /std:c++17 lib.cpp -I$pythonInclude $pybindIncludes /link /DLL /LIBPATH:C:\Users\bbrangeon\miniforge3\envs\pymdu\libs /Fe:rasterize$extSuffix
"@

echo $compileCommand
# Lancer vcvars64.bat dans un sous-processus avec la commande
Write-Host "Initialisation de l'environnement Visual Studio..."
cmd /c "`"$vcvarsPath`" && $compileCommand"