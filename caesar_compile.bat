@echo off
py regcomp2.py ./caesar.rasm caesar.pasm --target=pASM
py regcomp2.py ./caesar.rasm diff_caesar.p --target=pASM.c
py regcomp2.py ./diff_caesar.p diff_caesar.pasm --target=pASM
py regcomp2.py caesar.pasm ocaesar.p --target=pASM.c
py regcomp2.py ./ocaesar.p ocaesar.pasm --target=pASM
pause
