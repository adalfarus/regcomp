@echo off
py regcomp.py ./example.rasm -o=out_example.asm
py regcomp.py ./example_big.rasm -o=out_example_big.asm
py regcomp.py ./best_example.rasm -o=out_best_example.asm
pause
