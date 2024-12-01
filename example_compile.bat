@echo off
py regcomp2.py --target=pASM ./example.rasm -o=out_example.pasm --logging-mode=INFO
py regcomp2.py --target=pASM ./example_big.rasm -o=out_example_big.pasm --logging-mode=INFO
py regcomp2.py --target=pASM ./best_example.rasm -o=out_best_example.pasm --logging-mode=INFO
pause
