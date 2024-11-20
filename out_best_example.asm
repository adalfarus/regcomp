0 lda #0
1 lda #1
2 sta 15
3 lda #6
4 add 15
5 sta 15
6 jmp 8
7 stp
8 lda #12
9 sta 14
10 lda #10
11 add 14
12 sta 16
13 jmp (15)
14 0
15 0
16 0
