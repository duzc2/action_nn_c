echo 5 5 > temp.txt
echo 0 >> temp.txt
echo 0 >> temp.txt
echo 0 >> temp.txt
echo 4 >> temp.txt
demo\move\infer\build\Debug\move_infer.exe < temp.txt
del temp.txt
