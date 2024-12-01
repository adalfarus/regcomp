@echo off
py regcomp2.py ./caesar.rasm ocaesar.pasm --target=pASM
py regcomp2.py ./caesar.rasm ocaesar.p --target=pASM.c
py regcomp2.py ./ocaesar.p ocaesar.pasm --target=pASM
pause
