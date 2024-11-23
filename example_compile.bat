@echo off
py regcomp2.py --target=pASM ./example.rasm -o=out_example.asm --logging-mode=INFO
py regcomp2.py --target=pASM ./example_big.rasm -o=out_example_big.asm --logging-mode=INFO
py regcomp2.py --target=pASM ./best_example.rasm -o=out_best_example.asm --logging-mode=INFO
pause
