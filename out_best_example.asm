0 lda #97
1 lda #1
2 lda #2
3 sta 16
4 lda #6
5 add 16
6 sta 16
7 jmp 9
8 stp 
9 lda #12
10 sta 15
11 lda #10
12 add 15
13 sta 17
14 jmp (16)
15 0
16 0
17 0
100 97
