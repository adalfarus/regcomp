0 lda #97
1 lda #1
2 lda #2
3 sta 18
4 lda #6
5 add 18
6 sta 18
7 jmp 10
8 jmp 2
9 stp 
10 lda #12
11 sta 17
12 lda #10
13 add 17
14 sta 19
15 jze 10
16 jmp (18)
17 0
18 0
19 0
100 97
